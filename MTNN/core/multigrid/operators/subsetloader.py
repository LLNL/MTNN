"""Holds Subsetloaders.

A Subsetloader is used to collect a set of minibatches of training
data to be used as a dataloader during the following training
cycle. Used e.g. at the beginning of each V Cycle.

"""
# standard
from abc import ABC, abstractmethod

# local
from MTNN.utils import logger, printer, deviceloader

log = logger.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['WholeSetLoader, NextKLoader']


####################################################################
# Interface
###################################################################
class _BaseSubsetLoader(ABC):
    """Base class for all Subsetloaders.

    A Subsetloader will collect some minibatches of training data and
    then generate a dataloader using just those minibatches. Used to
    collect a subset of data to focus on e.g. during a V-cycle.
    """
    
    """Overwrite this"""
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def get_subset_dataloader(self, dataloader):
        """Collect some minibatches and create a dataloader to be used during
        the next cycle.
        """
        raise NotImplementedError


###################################################################
# Implementations
####################################################################
class NextKLoader(_BaseSubsetLoader):
    """Collect the next k minibatches of data, possibly reordered.
    """
    
    def __init__(self, num_minibatches) -> None:
        """
        Create a dataloader focused on just the next k minibatches
        """
        super().__init__()
        self.num_minibatches = num_minibatches
        self.curr_ind = 0

    def get_subset_dataloader(self, dataloader):
        nextksize = self.num_minibatches * dataloader.batch_size
        if self.curr_ind + nextksize <= len(dataloader.dataset):
            indices = list(range(self.curr_ind, self.curr_ind + self.num_minibatches * dataloader.batch_size))
            self.curr_ind += nextksize
            if self.curr_ind >= len(dataloader.dataset):
                self.curr_ind = 0
        else:
            amount_from_front = nextksize - (len(dataloader.dataset) - self.curr_ind)
            indices = list(range(self.curr_ind, len(dataloader.dataset))) + list(range(0, amount_from_front))
            self.curr_ind = amount_from_front
        return torch.utils.data.DataLoader(dataloader.dataset, batch_size = dataloader.batch_size,
                                           sampler=indices)

    
class WholeSetLoader(_BaseSubsetLoader):
    def __init__(self) -> None:
        """
        Collects whole data set.
        """
        super().__init__()

    def get_subset_dataloader(self, dataloader):
        return dataloader
