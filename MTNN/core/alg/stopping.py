"""
Holds Multigrid stopping measures
"""
from abc import ABC, abstractmethod
# Functions or Class?


class BaseStopper(ABC):
    """
    Base Stopper Class
    * Overwrite this.
    """

    def __init__(self, epochs):
        self.epochs = epochs

    @abstractmethod
    def should_stop(self, **kwargs) -> bool:
        """
        Returns True when the training should stop
        Args:
            **kwargs:

        Returns:

        """
        raise NotImplementedError


class EpochStopper(BaseStopper):
    def __init__(self, epochs):
        super().__init__(epochs)
        self.epoch_count = 0

    def should_stop(self) -> bool:
        if self.epochs == self.epoch_count:
            return True

    def track(self):
        self.epoch_count = self.epoch_count + 1

    def reset(self):
        self.epoch_count = 0


class CycleStopper(BaseStopper):
    def __init__(self, epochs):
        pass

    def should_stop(self):
        pass
        # TODO: Fill in


class TotalLossStopper(BaseStopper):
    def __init__(self, epochs):
        pass


    def should_stop(self):
        pass
        # TODO: Fill in


class RelativeLossStopper(BaseStopper):
    def __init__(self, epochs):
        self.epochs = epochs

    def should_stop(self):
        pass
        # TODO: Fill in




"""
class Stopper:

    def __init__(self):
        self.TotalLoss = None
        self.RelativeLoss = None
        self.NumCycles = None
        self.NumEpochs = None


def ByTotalLoss():
    pass
    # TODO: Fill in


def ByRelativeLoss():
    pass
    # TODO: Fill in


def ByNumIterations():
    pass
    # TODO: Fill in


def ByNumEpochs(num=0):
    stopper = Stopper()
    stopper.NumEpochs = num
    return stopper
"""
