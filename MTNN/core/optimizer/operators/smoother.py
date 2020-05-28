"""
Holds Multigrid Smoothers
"""
# PyTorch
import torch.optim as optim
# local
import MTNN.utils.logger as logger

log = logger.get_logger(__name__, write_to_file =True)

####################################################################
# API
###################################################################
class BaseSmoother:
    """
    Base Training Algorithm Smoother
    * Overwrite this.
    """

    def apply(self, model, data, stopper, verbose: bool):
        raise NotImplementedError

###################################################################
# Implementation
####################################################################
class SGDSmoother(BaseSmoother):

    def __init__(self, model, loss, lr=0.01, momentum=0.9, log_interval=0):
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
            stopper: <mg.core.alg.optimizer.stopping> Criteria to determine when to stop applying smoother
            verbose: Prints statistics/output to standard out

        Returns:
            None

        """

        # TODO: Check Stoppers
        # TODO: Fix logging

        while not stopper.should_stop():
            for epoch in range(stopper.max_epochs):

                if verbose:
                    if hasattr(stopper, 'cycle_count'):
                        log.info(f"Cycle {stopper.cycle_count}/{stopper.max_cycles}")
                    log.info(f"Epoch {epoch + 1}/{stopper.max_epochs}")


                for batch_idx, data in enumerate(dataloader, 0):
                    # Show status bar
                    """
                    if verbose:   
                        total_work = len(dataloader)
                        logger.progressbar(batch_idx, total_work, status = "Training")
                    """
                    inputs, labels = data

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    outputs = model(inputs)
                    loss = self.loss(outputs, labels)

                    # Backward
                    loss.backward()
                    self.optimizer.step()

                    if verbose and (batch_idx % self.log_interval) == 0:
                        log.info(f"Epoch: {epoch}\t{batch_idx}/ {len(dataloader) * dataloader.batch_size}\t\tLoss: {loss.item()}")

                stopper.track()
        stopper.reset()



