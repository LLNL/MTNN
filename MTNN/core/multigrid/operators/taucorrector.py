"""
Holds Tau correction definitions
"""
# standard
from abc import ABC, abstractmethod
import copy

# PyTorch
import torch

# local
import MTNN.utils.logger as log

log = log.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['NullTau',
           'WholeSetTau',
           'MinibatchTau']


def put_tau_together(fine_tau, fine_grad, coarse_grad, ops):
    """ Numerical work to construct the tau correction vector.

    Inputs:
    fine_tau (ParamVector) - The tau correction vector from the next-finer level. [f^h]
    fine_grad (ParamVector) - The gradient from the next-finer level. [A^h(u)]
    coarse_grad (ParamVector) - The gradient from the current, coarse level. [A^{2h}(R*u)]
    ops (TransferOps) - The transfer operators from the fine to the current, coarse level.

    Output:
    (ParamVector) - The tau correction vector for the current, coarse level.
    """
    # Construct [f^h - A^h(u)]
    diff_params = fine_tau - fine_grad

    # Apply restriction to construct R * [f^h - A^h(u)]
    coarse_diff_params = ops @ diff_params

    # Add final term to construct R * [f^h - A^h(u^h)] + A^{2h}(R*u^h)
    return coarse_diff_params + coarse_grad
        

###################################################################
# Interface
####################################################################
class _BaseTauCorrector(ABC):
    """Overwrite this"""
    def __init__(self, loss_fn, gradient_extractor):
        """
        Attributes:
            loss_fn: <torch.nn.modules.loss> Loss function
            rhs_W: residual weights
            rhs_B: residual biases
        """
        self.loss_fn = loss_fn
        self.gradient_extractor = gradient_extractor

    @abstractmethod
    def get_fine_tau(self, batch_idx = None, mini_dataloader = None):
        """ Get the tau computed at this level for a given minibatch."""
        raise NotImplementedError

    @abstractmethod
    def compute_tau(self, coarse_level, fine_level, dataloader, operators, **kwargs):
        """Computes residual tau of the coarse-level"""
        raise NotImplementedError

    @abstractmethod
    def correct(self, model, loss, num_batches, **kwargs):
        """Returns corrected loss """
        raise NotImplementedError

###################################################################
# Implementation
####################################################################
class NullTau(_BaseTauCorrector):
    def __init__(self, loss_fn, gradient_extractor):
        super().__init__(loss_fn, gradient_extractor)

    def get_fine_tau(self, batch_idx = None, mini_dataloader = None):
        return 0.0

    def compute_tau(self, coarse_level, fine_level, dataloader, operators):
        pass

    def correct(self, model, loss, batch_idx, num_batches, verbose = False):
        pass

class WholeSetTau(_BaseTauCorrector):
    def __init__(self, loss_fn, gradient_extractor):
        super().__init__(loss_fn, gradient_extractor)
        self.tau = None

    def get_fine_tau(self, batch_idx = None, mini_dataloader = None):
        if self.tau is None:
            return 0.0
        else:
            return self.tau

    def compute_tau(self, coarse_level, fine_level, dataloader, operators, verbose=False):
        fine_tau = fine_level.corrector.get_fine_tau() # of type ParamVector
        fine_grad = self.gradient_extractor.extract_from_network(fine_level, dataloader, self.loss_fn) #ParamVector
        coarse_grad = self.gradient_extractor.extract_from_network(coarse_level, dataloader, self.loss_fn)
        # We're breaking encapsulation of the gradient extractor to
        # get at the converter underneath. Perhaps this suggests a refactor should happen.
        self.tau = put_tau_together(fine_tau, fine_grad, coarse_grad, operators)
        self.tau_network_format = copy.deepcopy(self.tau)
        self.gradient_extractor.converter.convert_MTNN_format_to_network(self.tau_network_format)

        
    def correct(self, model, loss, batch_idx, num_batches, verbose=False):
        if self.tau is not None:
            for layer_id in range(len(model.layers)):
                loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].weight, self.tau_network_format.weights[layer_id]))
                loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].bias, self.tau_network_format.biases[layer_id].reshape(-1)))


class MinibatchTau(_BaseTauCorrector):
    """A tau corrector that computes a tau correction for each minibatch,
    and cycles through the corrections one at a time.
    """

    def __init__(self, loss_fn, gradient_extractor):
        super().__init__(loss_fn, gradient_extractor)
        self.tau_array = None
        self.finer_level_corrector = None

    def get_fine_tau(self, batch_idx, mini_dataloader):
        """In general we won't have already computed the tau for this minibatch,
        so recursively go down to the finer levels to build up tau."""
        if self.finer_level_corrector is None:
            return 0.0
        else:
            return self.finer_level_corrector.compute_tau_for_one_minibatch(mini_dataloader)

    def compute_tau_for_one_minibatch(self, fine_level, coarse_level, batch_idx, mini_dataloader):
        fine_tau = self.fine_level.corrector.get_fine_tau(batch_idx, mini_dataloader)
        fine_grad = self.gradient_extractor.extract_from_network(fine_level, mini_dataloader, self.loss_fn)
        coarse_grad = self.gradient_extractor.extract_from_network(coarse_level, mini_dataloader, self.loss_fn)
        # fine_grad = self.fine_level.net.getGrad(mini_dataloader, self.loss_fn)
        # coarse_grad = self.coarse_level.net.getGrad(mini_dataloader, self.loss_fn)
        tau = put_tau_together(fine_tau, fine_grad, coarse_grad, self.operators)
        self.gradient_extractor.corrector.convert_MTNN_format_to_network(tau)
        return tau

    def compute_tau(self, coarse_level, fine_level, dataloader, operators, verbose=False):
        # Ensure the coarse corrector can reach back to this one,
        # needed for recursion when the hierarchy gets deeper than 2
        # levels
        self.finer_level_corrector = self
        # Storing operators also needed for recursion
        self.operators = operators

        # Clear self.tau_array for next sequence of minibatch taus by making new dict
        self.tau_array = {}
        for batch_idx, mini_batch_data in enumerate(dataloader):
            mini_dataloader = (mini_batch_data,)
            self.tau_array[batch_idx] = self.compute_tau_for_one_minibatch(mini_dataloader)

    def correct(self, model, loss, batch_idx, num_batches, verbose=False):
        if self.tau_array is not None:
            for layer_id in range(len(model.layers)):
                loss -= torch.sum(torch.mul(model.layers[layer_id].weight, self.tau_array[batch_idx].weights[layer_id]))
                loss -= torch.sum(torch.mul(model.layers[layer_id].bias, self.tau_array[batch_idx].biases[layer_id]))
            






