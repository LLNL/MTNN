"""
Holds Smoothers
"""
# PyTorch
import torch.optim as optim
# local
import MTNN.utils as utils



logger = utils.get_logger(__name__, create_file=True)


class BaseSmoother:
    """
    Training Algorithm Smoother
    """

    def apply(self, model, data, stopper, verbose: bool):
        raise NotImplementedError


class SGDSmoother(BaseSmoother):

    def __init__(self, model, loss, lr=0.01, momentum=0.9, log_interval=0):
        super(SGDSmoother, self).__init__()
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

        # TODO: Check Stopper only has one mechanism/attribute "activated"

        # By Number of Epochs
        if stopper.NumEpochs:
            for epoch in range(stopper.NumEpochs):
                print(f"Epoch {epoch + 1}/{stopper.NumEpochs}")

                for batch_idx, data in enumerate(dataloader, 0):
                    # Show status bar
                    if verbose:
                        total_work = len(dataloader)
                        utils.progressbar(batch_idx, total_work, status = "Training")

                    inputs, labels = data

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    outputs = model(inputs)
                    loss = self.loss(outputs, labels)

                    # Backward
                    loss.backward()
                    self.optimizer.step()

                    if (batch_idx % self.log_interval) == 0:
                        #print("Finished 10")
                        #print(f"Epoch: {epoch} {batch_idx * len(data)}/ {len(dataloader)}\t Loss: {loss.item()}")
                        logger.info(f"Epoch: {epoch} {batch_idx * len(data)}/ {len(dataloader)}\t Loss: {loss.item()}")



