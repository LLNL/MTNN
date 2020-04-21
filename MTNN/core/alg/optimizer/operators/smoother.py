"""
Holds Smoothers
"""
import torch.nn as nn
import torch.optim as optim


class BaseSmoother:
    """
    Training Algorithm Smoother
    """

    def apply(self, model, data, stopping):
        raise NotImplementedError


class SGDSmoother(BaseSmoother):

    def __init__(self, model, loss, lr=0.01, momentum=0.9):
        super(SGDSmoother, self).__init__()
        self.loss = loss
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    def apply(self, model, data, stopper, stdout=False):
        """
        Apply forward pass and backward pass to the model until stopping criteria is met.
        Args:
            model:
            data:
            stopper: stopping criteria
            stdout: Prints statistics/output to standard out

        Returns:

        """
        inputs, labels = data

        # TODO: Add stopping mechanism

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward
        outputs = model(inputs)
        loss = self.loss(outputs, labels)

        # Backward
        loss.backward()
        self.optimizer.step()







