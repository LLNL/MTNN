"""
Holds Multigrid Smoothers
"""
# standard
from abc import ABC, abstractmethod

# torch
import torch.optim as optim

# local
from MTNN.core.components import models, data
from MTNN.core.alg import stopping
from MTNN.utils import logger, printer, deviceloader

log = logger.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['SGDSmoother']

####################################################################
# Interface
###################################################################
class _BaseSmoother(ABC):
    """Overwrite this"""
    @abstractmethod
    def apply(self, model: models._BaseModel, dataloader: data._BaseDataLoader,
              stopper: stopping._BaseStopper, tau, verbose: bool):
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class SGDSmoother(_BaseSmoother):
    def __init__(self, model, loss_fn, optim_params, stopper,  log_interval=0) -> None:
        """
        "Smooths" the error by applying Stochastic Gradient Descent on the model
        Args:
            model:  <core.components.models.BaseModel>
            loss_fn:  <torch.nn.modules.loss> Instance of a PyTorch loss function
            optim_params: <collections.namedtuple> Named Tuple of optimizer parameters
            log_interval: <int> Controls frequency (every # of minibatches) to log
            stopper: <core.alg.stopping> Stopper
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=optim_params.lr,
                                   momentum=optim_params.momentum,
                                   weight_decay=optim_params.l2_decay)
        self.stopper = stopper
        self.log_interval = log_interval

    def apply(self, model, dataloader, stopper, tau=None, verbose=False) -> None:
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
        # TODO: refactor with pipeline/token?
        # TODO: Refactor Stoppers
        # TODO: Fix logging
        while not self.stopper.should_stop():
            for epoch in range(stopper.max_epochs):
                for batch_idx, mini_batch_data in enumerate(dataloader, 0):
                    input_data, target_data = deviceloader.load_data(mini_batch_data, model.device)
                    self.loss_fn.to(model.device)
                   
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    # Forward
                    outputs = model(input_data)
#                    printer.printModel(model, msg="Smoother.Before loss",val= True)
                    loss = self.loss_fn(outputs, target_data)
#                    printer.printModel(model, msg="Smoother.After loss", val = True)
#                   log.debug(f"Smoother.Loss: {loss}")
                    #printer.printModel(model, val = True, grad = True)

                    # Apply Tau Correction
                    if tau:
                        tau.correct(model, len(dataloader), loss, verbose)

                    # Backward
                    loss.backward()
#                    printer.printModel(model, msg = "After loss.backward", val = True)

                    self.optimizer.step()
                    if verbose:
                        # Show status bar
                        #total_work = len(dataloader)
                        #logger.progressbar(batch_idx, total_work, status = "Training")
                        printer.print_smoother(epoch, loss, batch_idx, dataloader, stopper, self.log_interval)
#                      printer.print_model(model, msg= "Smoother.After optimizer update", val = True)
                self.stopper.track()
        self.stopper.reset()










