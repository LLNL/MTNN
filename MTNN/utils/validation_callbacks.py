import torch
import torch.nn as nn
from MTNN.utils import deviceloader

#=====================================
# Loss and loss accumulator functions
#=====================================

linf_loss = lambda outputs, true_outputs : torch.max (torch.max(torch.abs(true_outputs - outputs), dim=1).values)

class classifier_inaccuracy:
    def __init__(self, num_batches):
        self.num_batches = num_batches
        
    def __call__(self, outputs, labels):
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        num_incorrect = (predicted != labels).sum().item()
        return float(num_incorrect) / (self.num_batches * total)

sum_accumulator = lambda x, y : x + y
max_accumulator = lambda x, y : torch.max(x, y)


#============================================
# Validation callback functions for reporting
#============================================

class ValidationCallback:
    """After each training iteration, a callback is called for computing
    validation losses over time. This is a general function that
    accepts a set of loss functions to report on, and computes the
    losses over the validation set for each level of the multilevel
    hierarchy.

    """
    
    def __init__(self, val_dataloader,
                 loss_fns, accumulator_fns, loss_names,
                 num_levels,
                 test_frequency = 1):
        """ ValidationCallback constructor.

        @param val_dataloader Validation dataloader
        @param loss_fns List of loss functions
        @param accumulator_fns List of functions used to accumulate loss over minibatches
        @param loss_names List of names used for loss functions
        @param num_levels Number of levels in the hierarchy
        @param test_frequency Only report every test_frequency iterations
        """
        self.val_dataloader = val_dataloader
        self.loss_fns = loss_fns
        self.accumulator_fns = accumulator_fns
        self.loss_names = loss_names
        self.test_frequency = test_frequency

        self.num_losses = len(self.loss_fns)
        self.num_levels = num_levels
        self.best_seen = float('inf') * torch.ones((self.num_levels, self.num_losses))

    def __call__(self, levels, cycle):
        if type(cycle) != str and (cycle + 1) % self.test_frequency != 0:
            return

        for level in levels:
            level.net.eval()

        with torch.no_grad():
            total_losses = torch.zeros((self.num_levels, self.num_losses))
            curr_losses = torch.zeros((self.num_levels, self.num_losses))
            for mini_batch_data in self.val_dataloader:
                inputs, true_outputs  = deviceloader.load_data(mini_batch_data, levels[0].net.device)
                for level_ind, level in enumerate(levels):
                    outputs = level.net(inputs)
                    curr_losses[level_ind,:] = \
                        torch.tensor([self.loss_fns[i](outputs, true_outputs) for i in range(self.num_losses)])
                for i in range(self.num_losses):
                    total_losses[:,i] = self.accumulator_fns[i](total_losses[:,i], curr_losses[:,i])

            self.best_seen = torch.min(total_losses, self.best_seen)

            for level_ind in range(self.num_levels):
                loss_str = ", ".join(["{} loss {:.5f} (best seen {:.5f})".
                                      format(self.loss_names[i], total_losses[level_ind, i],
                                             self.best_seen[level_ind, i]) for i in range(self.num_losses)])
                print("Level {}, Cycle {}: {}".format(level_ind, cycle, loss_str), flush=True)

        for level in levels:
            level.net.train()



class RealValidationCallback(ValidationCallback):
    """ Validation callback class useful for real-valued output data.
    """
    def __init__(self, val_dataloader, num_levels, test_frequency = 1):
        super().__init__(val_dataloader,
                         [nn.MSELoss(reduction="sum"), linf_loss],
                         [sum_accumulator, max_accumulator],
                         ["L2", "Linf"],
                         num_levels, test_frequency)


class ClassifierValidationCallback(ValidationCallback):
    """ Validation callback class useful for classification data.
    """
    def __init__(self, val_dataloader, num_levels, test_frequency = 1):
        super().__init__(val_dataloader,
                         [nn.CrossEntropyLoss(), classifier_inaccuracy],
                         [sum_accumulator, sum_accumulator],
                         ["cross entropy", "accuracy"],
                         num_levels, test_frequency)
