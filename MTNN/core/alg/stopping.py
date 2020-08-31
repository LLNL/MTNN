"""
Holds Multigrid stopping measures
"""
# standard
from abc import ABC, abstractmethod

# Public
__all__ = ['EpochStopper',
           'CycleStopper']


###################################################################
# Interface
####################################################################
class _BaseStopper(ABC):
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

###################################################################
# Implementation
####################################################################
class EpochStopper(_BaseStopper):
    """
    Stops the training loop based on epoch count
    """
    def __init__(self, epochs):
        super().__init__(epochs)
        self.epoch_count = 0

    def should_stop(self) -> bool:
        if self.epoch_count == self.max_epochs:
            return True
        else:
            return False

    def track(self):
        self.epoch_count += 1

    def reset(self):
        self.epoch_count = 0


class CycleStopper(_BaseStopper):
    """
    Stops the training loop based on number of cycles through a multigrid hierarchy
    # TODO: Fix this
    """
    def __init__(self, epochs, cycles):
        super().__init__(epochs)
        self.max_cycles = cycles
        self.cycle_count = 0

    def track(self):
        self.cycle_count += 1

    def should_stop(self):
        if self.cycle_count == self.max_cycles:
            print("Stopping...")
            return True
        else:
            return False

    def reset(self):
        self.cycle_count = 0


class TotalLossStopper(_BaseStopper):
    def __init__(self, epochs):
        pass

    def should_stop(self):
        pass
        # TODO: Fill in


class NoLongerImprovingStopper(_BaseStopper):
    """
    Stops the training loop if the loss stops improving within the epoch
    """
    def __init__(self, epochs):
        self.epochs = epochs

    def should_stop(self):
        pass
        # TODO: Fill in


