import torch
import torch.nn as nn
import torch.nn.functional as F
from MTNN.utils import deviceloader, logger
from abc import ABC, abstractmethod

#=====================================
# Loss functions
#=====================================

mse_loss = lambda outputs, true_outputs : torch.sum(F.mse_loss(outputs, true_outputs, reduction="none"), dim=1)
linf_loss = lambda outputs, true_outputs : torch.max(torch.abs(true_outputs - outputs), dim=1).values
def inaccuracy(outputs, labels):
    num_labels= labels.size(0)
    _, predicted = torch.max(outputs.data, 1)
    return predicted != labels

#=====================================
# Loss accumulators
#=====================================

class Accumulator:
    def __init__(self, num_levels):
        self.num_levels = num_levels
        self.reset()

    @abstractmethod
    def accumulate(self, level_ind, loss):
        """
        @param level_ind The level of the hierarchy
        @param loss A list or first-order tensor containing the loss for each example in this batch
        """
        raise NotImplementedError

    def reset(self):
        self.accumulated_losses = [0.0] * self.num_levels
        
    def get_accumulated_losses(self):
        return self.accumulated_losses

class SumAccumulator(Accumulator):
    def __init__(self, num_levels):
        super().__init__(num_levels)

    def accumulate(self, level_ind, loss):
        self.accumulated_losses[level_ind] += torch.sum(loss).item()

class MeanAccumulator(Accumulator):
    def __init__(self, num_levels):
        super().__init__(num_levels)
        self.reset()

    def reset(self):
        super().reset()
        self.examples_seen = [0] * self.num_levels

    def accumulate(self, level_ind, loss):
        self.accumulated_losses[level_ind] += torch.sum(loss).item()
        self.examples_seen[level_ind] += len(loss)

    def get_accumulated_losses(self):
        return [self.accumulated_losses[i] / self.examples_seen[i] for i in range(self.num_levels)]
    
class MaxAccumulator(Accumulator):
    def __init__(self, num_levels):
        super().__init__(num_levels)

    def accumulate(self, level_ind, loss):
        self.accumulated_losses[level_ind] = max(self.accumulated_losses[level_ind], torch.max(loss).item())

#============================================
# Validation callback functions for reporting
#============================================

class ValidationCallback:
    """After each training iteration, a callback is called for computing
    validation losses over time. This is a general function that
    accepts a set of loss functions to report on, and computes the
    losses over the validation set for each level of the multilevel
    hierarchy.

    Each loss must produce a 1st-order tensor of lenght equal to the
    number of examples given as input. This tensor is then fed to an
    associated Accumulator, which collects all the reported losses and
    then summarizes them.

    This class assumes that lower is better, so if you want to report
    on something in which higher is better (e.g. classifier accuracy),
    report on its inverse (e.g. classifier inaccuracy).

    """
    
    def __init__(self, val_dataloader,
                 loss_fns, accumulators, loss_names,
                 num_levels, val_frequency = 1):
        """ ValidationCallback constructor.

        @param val_dataloader Validation dataloader
        @param loss_fns List of loss functions
        @param accumulators List of accumulators used to accumulate loss over minibatches
        @param loss_names List of names used for loss functions
        @param num_levels Number of levels in the hierarchy
        @param test_frequency Only report every test_frequency iterations
        """
        self.val_dataloader = val_dataloader
        self.loss_fns = loss_fns
        self.accumulators = accumulators
        self.loss_names = loss_names
        self.val_frequency = val_frequency

        self.num_losses = len(self.loss_fns)
        self.num_levels = num_levels
        self.best_seen = float('inf') * torch.ones((self.num_levels, self.num_losses))
        self.log = logger.get_MTNN_logger()

    def __call__(self, levels, cycle = None):
        if cycle is not None and (cycle + 1) % self.val_frequency != 0:
            return

        for level in levels:
            level.net.eval()

        with torch.no_grad():
            [acc.reset() for acc in self.accumulators]
            for mini_batch_data in self.val_dataloader:
                inputs, true_outputs  = deviceloader.load_data(mini_batch_data, levels[0].net.device)
                for level_ind, level in enumerate(levels):
                    outputs = level.net(inputs)
                    [self.accumulators[i].accumulate(level_ind, loss(outputs, true_outputs))
                     for i, loss in enumerate(self.loss_fns)]

            total_losses = torch.Tensor([acc.get_accumulated_losses() for acc in self.accumulators]).T
            self.best_seen = torch.min(self.best_seen, total_losses)
                                       
            for level_ind in range(self.num_levels):
                loss_str = ", ".join(["{} {:.5e} (best seen {:.5e})".
                                      format(self.loss_names[i], total_losses[level_ind, i],
                                             self.best_seen[level_ind, i]) for i in range(self.num_losses)])
                cycle_str = "Cycle {}".format(cycle+1) if cycle is not None else "Finished"
                self.log.warning("Level {}, {}: {}".format(level_ind, cycle_str, loss_str))

        for level in levels:
            level.net.train()



class RealValidationCallback(ValidationCallback):
    """ Validation callback class useful for real-valued output data.
    """
    def __init__(self, val_dataloader, num_levels, test_frequency = 1):
        super().__init__(val_dataloader,
                         [mse_loss, linf_loss],
                         [MeanAccumulator(num_levels), MaxAccumulator(num_levels)],
                         ["L2 loss", "Linf loss"],
                         num_levels, test_frequency)


class ClassifierValidationCallback(ValidationCallback):
    """ Validation callback class useful for classification data.
    """
    def __init__(self, val_dataloader, num_levels, val_frequency = 1):
        super().__init__(val_dataloader,
                         [nn.CrossEntropyLoss(), inaccuracy],
                         [SumAccumulator(num_levels), MeanAccumulator(num_levels)],
                         ["cross entropy loss", "fraction incorrect"],
                         num_levels, val_frequency)
