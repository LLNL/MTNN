"""Holds Subsetloaders.

A Subsetloader is used to collect a set of minibatches of training
data to be used as a dataloader during the following training
cycle. Used e.g. at the beginning of each V Cycle.

"""
# standard
from abc import ABC, abstractmethod

import torch

# Public
__all__ = ['WholeSetLoader',
           'NextKLoader',
           'CyclingNextKLoader']


####################################################################
# Interface
###################################################################
class BaseSubsetLoader(ABC):
    """Base class for all Subsetloaders.

    A Subsetloader will collect some minibatches of training data and
    then generate a dataloader using just those minibatches. Used to
    collect a subset of data to focus on during a V-cycle.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_subset_dataloader(self, dataloader):
        """
        Collect some minibatches and create a dataloader to be used during
        the next cycle.
        """
        raise NotImplementedError

###################################################################
# Implementations
####################################################################
class WholeSetLoader(BaseSubsetLoader):
    def __init__(self):
        """Collects whole data set. Using this implies that each smoothing
        pass will execute one or more epochs, and that the WholeSetTau
        will compute its tau correction over the entire training set.

        """
        super().__init__()

    def get_subset_dataloader(self, dataloader):
        return dataloader

class NextKLoader(BaseSubsetLoader):
    """
    Collect the next k minibatches data, possibly reordered.
    """
    
    def __init__(self, num_minibatches, shuffle=False):
        """
        Create a dataloader focused on just the next k minibatches
        """
        super().__init__()
        self.num_minibatches = num_minibatches
        self.curr_ind = 0
        self.shuffle = shuffle

    def get_subset_dataloader(self, dataloader):
        """Get next k minibatches of data. If there are not k minibatches
        left in the dataset, take whatever is left, and then add to it
        the first k minibatches in the dataset. Thus, this will
        occasionally produce a larger dataset, but will ensure we use
        the same datasets each epoch.

        """
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
            indices = list(range(self.curr_ind, len(dataloader.dataset))) + list(range(0, nextksize))
            self.curr_ind = nextksize
        subdataset = torch.utils.data.Subset(dataloader.dataset, indices)
        return torch.utils.data.DataLoader(
            subdataset, batch_size = dataloader.batch_size, shuffle=self.shuffle)

class CyclingNextKLoader(NextKLoader):
    """Like the NextKLoader, but changes how many minibatches we draw
    each time.

    """

    def __init__(self, num_minibatch_array):
        """@param num_minibatch_array list of minibatch sizes to draw. It is
        recommended to make a list length that is a multiple of $2 *
        num_hierarchy_levels - 1$ so that it will align with the
        V-cycle

        """
        super().__init__(num_minibatch_array[0])
        self.num_minibatch_array = num_minibatch_array
        self.minibatch_array_ind = 0

    def get_subset_dataloader(self, dataloader):
        self.num_minibatches = self.num_minibatch_array[self.minibatch_array_ind]
        self.minibatch_array_ind = (self.minibatch_array_ind + 1) % len(self.num_minibatch_array)
        return super().get_subset_dataloader(dataloader)
