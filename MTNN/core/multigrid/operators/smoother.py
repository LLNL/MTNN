"""
Holds Multigrid Smoothers
"""
# standard
from abc import ABC, abstractmethod

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# local
from MTNN.core.components import models, data
from MTNN.core.alg import stopping
from MTNN.utils import logger, printer

log = logger.get_logger(__name__, write_to_file =True)

__all__ = ['SGDSmoother']
####################################################################
# Interface
###################################################################
class _BaseSmoother(ABC):
    """ Overwrite this"""
    @abstractmethod
    def apply(self, model: models, data: data, stopper: stopping, verbose: bool):
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class SGDSmoother(_BaseSmoother):

    def __init__(self, model: models, loss_fn: nn.modules.loss._Loss, optim_params,
                 stopper: stopping._BaseStopper, log_interval=0) -> None:
        """
        "Smooths" the error by applying Stochastic Gradient Descent on the model
        Args:
            model:  <core.components.models.BaseModel>
            loss_fn:  <torch.nn.modules.loss> Instance of a PyTorch loss function
            optim_params: <collections.namedtuple> Named Typle of optimizer parameters
            log_interval: <int> Indicates frequency (every # of minibatches) to log
            stopper: <core.alg.stopping._BaseStopper>
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=optim_params.lr,
                                   momentum=optim_params.momentum,
                                   weight_decay=optim_params.l2_decay)
        self.log_interval = log_interval
        self.stopper = stopper

    def apply(self, model, dataloader,  stopper, rhs=None,  verbose=False) -> None:
        #TODO: refactor with pipeline/token?
        """
        Apply forward pass and backward pass to the model until stopping criteria is met.
        Args:
            model: Class <core.components.models.BaseModel>
            dataloader: <torch.utils.data.DataLoader> PyTorch Dataloader 
            stopper: <mg.core.alg.multigrid.stopping> Criteria to determine when to stop applying smoother
            verbose: <bool> Prints statistics/output to standard out

        Returns:
            None
        """

        # TODO: Fix/Check about Stoppers
        # TODO: Fix logging
        # TODO: Apply SGD not batch gradient descent
        while not self.stopper.should_stop():
            for epoch in range(stopper.max_epochs):
                for batch_idx, mini_batch_data in enumerate(dataloader, 0):

                    # Show status bar
                    """
                    if verbose:   
                        total_work = len(dataloader)
                        logger.progressbar(batch_idx, total_work, status = "Training")
                    """
                    inputs, labels = mini_batch_data

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    outputs = model(inputs)
#                    printer.printModel(model, msg="Smoother.Before loss",val= True)
                    loss = self.loss_fn(outputs, labels)
#                    printer.printModel(model, msg="Smoother.After loss", val = True)

#                    log.debug(f"Smoother.Loss: {loss}")
                    #printer.printModel(model, val = True, grad = True)

                    # Backward
                    loss.backward()
                    printer.printModel(model, msg = "After loss.backward", val = True)
                    self.optimizer.step()
                    if verbose:
                        printer.printSmoother(epoch + 1, loss, batch_idx, dataloader, stopper, self.log_interval)
                        printer.printModel(model, msg="Smoother.After optimizer update", val = True)

        # Apply Tau Correction
        #if tau_correction:
            # TODO: Refactor with higher-order FAS class
            nbatches = len(dataloader)

            if rhs.W and rhs.b is not None:
                try:
                    rhsW_arr = [torch.from_numpy(A).to(device) for A in rhs.W]
                    rhsB_arr = [torch.from_numpy(A).to(device) for A in rhs.b]
                    for layer_id in range(len(model.layers)):
                        loss -= (1.0 / nbatches) * torch.sum(torch.mul(model.layers[layer_id].weight, rhsW_arr[layer_id]))
                        loss -= (1.0 / nbatches) * torch.sum(torch.mul(model.layers[layer_id].bias, rhsB_arr[layer_id]))
                except Exception as e:
                    raise

            self.stopper.track()
        self.stopper.reset()






