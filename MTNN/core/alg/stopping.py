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
        self.max_epochs = epochs

    @abstractmethod
    def should_stop(self, **kwargs) -> bool:
        """
        Returns True when the training should stop
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class EpochStopper(BaseStopper):
    """
    Stops the training loop based on epoch count
    """
    def __init__(self, epochs):
        super().__init__(epochs)
        self.epoch_count = 0

    def should_stop(self) -> bool:
        if self.epoch_count == self.max_epochs:
            return True

    def track(self):
        self.epoch_count += 1

    def reset(self):
        self.epoch_count = 0


class CycleStopper(BaseStopper):
    """
    Stops the training loop based on multigrid cycle count
    """
    def __init__(self, epochs):
        super().__init__(epochs)
        self.max_cycles = 0
        self.cycle_count = 0

    def track(self):
        self.cycle_count += 1

    def should_stop(self):
        if self.cycle_count == self.max_cycles:
            return True

    def reset(self):
        self.cycle_count = 0


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


