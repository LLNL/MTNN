"""
Holds Multigrid Smoothers
"""
# standard
from abc import ABC, abstractmethod

# torch
import torch
import torch.optim as optim

# local
from MTNN.utils import logger, printer, deviceloader
from MTNN.core.multigrid.operators.interpolator import transfer_star

log = logger.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['SGDSmoother']


####################################################################
# Interface
###################################################################
class _BaseSmoother(ABC):
    """Overwrite this"""
    def __init__(self, model, loss_fn, optim_params, log_interval=0) -> None:
        """
        Args:
            model:  <core.components.models.BaseModel>
            loss_fn:  <torch.nn.modules.loss> Instance of a PyTorch loss function
            optim_params: <collections.namedtuple> Named Tuple of optimizer parameters
            log_interval: <int> Controls frequency (every # of minibatches) to log
        """
        self.loss_fn = loss_fn
        self.optim_params = optim_params
        self.log_interval = log_interval
        self.optimizer = None
        self.momentum_data = None

    @abstractmethod
    def apply(self, model, dataloader, tau, verbose: bool):
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class SGDSmoother(_BaseSmoother):
    def __init__(self, model, loss_fn, optim_params, log_interval=0) -> None:
        """
        "Smooths" the error by applying Stochastic Gradient Descent on the model
        """
        super().__init__(model, loss_fn, optim_params, log_interval)

    def apply(self, model, dataloader, num_epochs, tau=None, l2_info = None, verbose=False) -> None:
        """
        Apply forward pass and backward pass to the model until stopping criteria is met.
        Optionally apply tau correction if tau_corrector is given.
        Args:
            model: Class <core.components.models> BaseModel
            dataloader: <torch.utils.data.DataLoader> PyTorch Dataloader 
            stopper: <core.alg.stopping> Criteria to determine when to stop applying smoother
            tau: <core.multigrid.operators.tau_correct> BaseTauCorrector
            verbose: <bool> Prints statistics/output to standard out

        Returns:
            None
        """
        if self.optimizer is None:
            self.optimizer = optim.SGD(model.parameters(),
                                       lr = self.optim_params.lr,
                                       momentum = self.optim_params.momentum,
                                       weight_decay = self.optim_params.l2_decay)
        if self.momentum_data is not None:
            # Insert momentum data
            for i in range(0, len(self.optimizer.param_groups[0]['params']), 2):
                self.optimizer.state[self.optimizer.param_groups[0]['params'][i]]['momentum_buffer'] = self.momentum_data[i]
                self.optimizer.state[self.optimizer.param_groups[0]['params'][i+1]]['momentum_buffer'] = self.momentum_data[i+1]
            self.momentum_data = None
            
        # TODO: Fix logging
        for epoch in range(num_epochs):
            for batch_idx, mini_batch_data in enumerate(dataloader):
                input_data, target_data = deviceloader.load_data(mini_batch_data, model.device)
                self.loss_fn.to(model.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                                
                # Forward
                outputs = model(input_data)
                loss = self.loss_fn(outputs, target_data)

                # # Apply l2 regularization
                # l2_term = 0.0
                # if l2_info is None:
                #     for layer in model.layers:
                #         l2_term += torch.sum(layer.weight * layer.weight)
                #         l2_term += torch.sum(layer.bias * layer.bias)
                #     loss += (self.optim_params.l2_decay / 2.0) * l2_term
                # else:
                #     zW, zB, l2_reg_left_vecs, l2_reg_right_vecs = l2_info
                #     weights = [layer.weight for layer in model.layers]
                #     biases = [layer.bias for layer in model.layers]
                #     otherW, otherB = transfer_star(weights, biases, l2_reg_left_vecs, l2_reg_right_vecs)
                #     for layer_idx, layer in enumerate(model.layers):
                #         l2_term += torch.sum(layer.weight * otherW[layer_idx])
                #         l2_term += torch.sum(layer.bias * otherB[layer_idx])
                #         l2_term += torch.sum(layer.weight * zW[layer_idx])
                #         l2_term += torch.sum(layer.bias * zB[layer_idx])
                #     loss += (self.optim_params.l2_decay / 2.0) * l2_term
                    
                # Apply Tau Correction
                if tau:
                    tau.correct(model, loss, batch_idx, len(dataloader), verbose)

                # Backward
                loss.backward()
                
                self.optimizer.step()
                if verbose:# and (batch_idx + 1) % 50 == 0:
                    # Show status bar
                    #total_work = len(dataloader)
                    #logger.progressbar(batch_idx, total_work, status = "Training")
                    printer.print_smoother(loss, batch_idx, dataloader, self.log_interval, tau)








