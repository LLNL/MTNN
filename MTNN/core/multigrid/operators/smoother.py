"""
Holds Multigrid Smoothers
"""
from abc import ABC, abstractmethod
# PyTorch
import torch.optim as optim
# local
from MTNN.utils import logger, printer

log = logger.get_logger(__name__, write_to_file =True)

####################################################################
# Interface
###################################################################
class BaseSmoother(ABC):
    """ Overwrite this"""
    @abstractmethod
    def apply(self, model, data, stopper, verbose: bool):
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class SGDSmoother(BaseSmoother):

    def __init__(self, model, loss, lr=0.01, momentum=0.9, log_interval=0):
        """

        Args:
            model: Class <core.components.models.BaseModel>
            loss:
            lr:
            momentum:
            log_interval:
        """
        super().__init__()
        self.loss = loss
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.log_interval = log_interval

    def apply(self, model, dataloader,  stopper, verbose=False):
        """
        Apply forward pass and backward pass to the model until stopping criteria is met.
        Args:
            model: PyTorch Neural network Object
            dataloader: PyTorch Dataloader Object
            stopper: <mg.core.alg.multigrid.stopping> Criteria to determine when to stop applying smoother
            verbose: Prints statistics/output to standard out

        Returns:
            None

        """


        # TODO: Fix/Check about Stoppers
        # TODO: Fix logging
        # TODO: Apply SGD not batch gradient descent
        while not stopper.should_stop():
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
                    loss = self.loss(outputs, labels)

                    # Backward
                    loss.backward()
                    self.optimizer.step()

                    if verbose:

                        printer.printSmoother(epoch + 1, loss, batch_idx, dataloader, stopper, self.log_interval)

                stopper.track()
        stopper.reset()






