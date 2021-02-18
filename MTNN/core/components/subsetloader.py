"""Holds Subsetloaders.

A Subsetloader is used to collect a set of minibatches of training
data to be used as a dataloader during the following training
cycle. Used e.g. at the beginning of each V Cycle.

"""
# standard
from abc import ABC, abstractmethod

# local
from MTNN.utils import logger, printer, deviceloader

import torch

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
        if (nextksize > len(dataloader.dataset)):
            raise ValueError("Trying to make a subset of size {} but whole dataset is only of size {}".
                             format(nextksize, len(dataloader.dataset)))
        if self.curr_ind + nextksize <= len(dataloader.dataset):
            indices = list(range(self.curr_ind, self.curr_ind + self.num_minibatches * dataloader.batch_size))
            self.curr_ind += nextksize
            if self.curr_ind >= len(dataloader.dataset):
                self.curr_ind = 0
        else:
            # amount_from_front = nextksize - (len(dataloader.dataset) - self.curr_ind)
            # indices = list(range(self.curr_ind, len(dataloader.dataset))) + list(range(0, amount_from_front))
            # self.curr_ind = amount_from_front
            indices = list(range(self.curr_ind, len(dataloader.dataset))) + list(range(0, nextksize))
            self.curr_ind = nextksize
        return torch.utils.data.DataLoader(dataloader.dataset, batch_size = dataloader.batch_size,
                                           sampler=torch.utils.data.SubsetRandomSampler(indices))

    
class WholeSetLoader(_BaseSubsetLoader):
    def __init__(self) -> None:
        """
        Collects whole data set.
        """
        super().__init__()

    def get_subset_dataloader(self, dataloader):
        return dataloader

class CyclingNextKLoader(NextKLoader):
    """ Like the NextKLoader, but change how many minibatches each time.
    """

    def __init__(self, num_minibatch_array) -> None:
        super().__init__(num_minibatch_array[0])
        self.num_minibatch_array = num_minibatch_array
        self.minibatch_array_ind = 0

    def get_subset_dataloader(self, dataloader):
        self.num_minibatches = self.num_minibatch_array[self.minibatch_array_ind]
        self.minibatch_array_ind += 1
        if self.minibatch_array_ind == len(self.num_minibatch_array):
            self.minibatch_array_ind = 0
        return super().get_subset_dataloader(dataloader)
