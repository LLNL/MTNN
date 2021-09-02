"""
Holds Tau correction definitions
"""
from abc import ABC, abstractmethod
import torch
import MTNN.utils.logger as log
import MTNN.utils.printer as printer
from MTNN.utils.datatypes import ParamVector, TransferOps
#import MTNN.core.multigrid.operators.interpolator as interp

import copy
import sys

log = log.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['BasicTau']

# TODO: remove
# def transfer(Wmats, Bmats, R_ops, P_ops):
#     # Matrix multiplication broadcasting:
#     # T of shape (a, b, c, N, M)
#     # Matrix A of shape (k, N)
#     # Matrix B of shape (M, p)
#     # A @ T @ B computes a tensor of shape (a, b, c, k, p) such that
#     # (i, j, l, :, :) = A @ T(i,j,l,:,:) @ B
#     num_layers = len(Wmats)
#     Wdest_array = []
#     Bdest_array = []
#     for layer_id in range(num_layers):
#         Wsrc = Wmats[layer_id]
#         Bsrc = Bmats[layer_id]
#         if layer_id < num_layers - 1:
#             if layer_id == 0:
#                 Wdest = R_ops[layer_id] @ Wsrc
#             else:
#                 Wdest = R_ops[layer_id] @ Wsrc @ P_ops[layer_id - 1]
#             Bdest = R_ops[layer_id] @ Bsrc
#         elif layer_id > 0:            
#             Wdest = Wsrc @ P_ops[layer_id-1]
#             Bdest = Bsrc.clone()

#         Wdest_array.append(Wdest)
#         Bdest_array.append(Bdest)
#     return Wdest_array, Bdest_array

# def put_tau_together(fine_tau, fine_grad, coarse_grad, ops):
#     """ Numerical work to construct the tau correction vector.

#     Inputs:
#     fine_tau (ParamVector) - The tau correction vector from the next-finer level. [f^h]
#     fine_grad (ParamVector) - The gradient from the next-finer level. [A^h(u)]
#     coarse_grad (ParamVector) - The gradient from the current, coarse level. [A^{2h}(R*u)]
#     ops (TransferOps) - The transfer operators from the fine to the current, coarse level.

#     Output:
#     (ParamVector) - The tau correction vector for the current, coarse level.
#     """
#     num_fine_layers = len(fine_tau.weights)
#     W_rhs_array, B_rhs_array = [], []

#     # Construct [f^h - A^h(u)]
#     for layer_id in range(num_fine_layers):
#         W_rhs_array.append(fine_tau.weights[layer_id] - fine_grad.weights[layer_id])
#         B_rhs_array.append(fine_tau.biases[layer_id] - fine_grad.biases[layer_id])

#     # Apply restriction to construct R * [f^h - A^h(u)]
#     W_c_rhs_array, B_c_rhs_array = transfer(W_rhs_array, B_rhs_array, ops.R_for_grad_op, ops.P_for_grad_op)

#     # Add final term to construct R * [f^h - A^h(u^h)] + A^{2h}(R*u^h)
#     for layer_id in range(num_fine_layers):
#         W_c_rhs_array[layer_id] += coarse_grad.weights[layer_id]
#         B_c_rhs_array[layer_id] += coarse_grad.biases[layer_id]

#     return ParamVector(W_c_rhs_array, B_c_rhs_array)

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
                loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].bias, self.tau_network_format.biases[layer_id]))


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

    def compute_tau_for_one_minibatch(self, mini_dataloader):
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
            


















# ###################################################################
# # Implementation
# ####################################################################
# class BasicTau(_BaseTauCorrector):
#     def __init__(self, loss_fn):
#         super().__init__(loss_fn)
#         self.tau = None

#     def get_fine_tau(self, batch_idx = None):
#         if self.rhs is None:
#             return 0.0
#         else:
#             return self.tau

#     def compute_tau(self, fine_level, coarse_level, dataloader, operators, verbose=False):
#         # When restricting from L0 to L1, the rhs of the problem is 0,
#         # but at coarser levels of the hierarchy, there is a nonzero
#         # rhs due to previous tau corrections. In that case, that
#         # nonzero rhs must be part of the correction as well.
#         fine_rhs = None
#         if fine_level.corrector.rhs_W:
#             fine_rhs = (fine_level.corrector.rhs_W, fine_level.corrector.rhs_B)

#         (rhsW, rhsB) = self.get_tau_for_data(fine_level, coarse_level, dataloader,
#                                              operators, self.loss_fn, fine_rhs)
#         coarse_level.corrector.rhs_W = rhsW
#         coarse_level.corrector.rhs_B = rhsB
        
#     def correct(self, model, loss, batch_idx, num_batches, verbose=False):
#         if self.rhs_W and self.rhs_B is not None:
#             try:
#                 for layer_id in range(len(model.layers)):
#                     loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].weight, self.rhs_W[layer_id]))
#                     loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].bias, self.rhs_B[layer_id]))

#                 if verbose:
#                     printer.print_tau(self, loss.item(), msg="\tApplying Tau Correction: ")
#             except Exception as e:
#                 print("Exception in tau_corrector.py:BasicTau.correct.", file=sys.stderr)
#                 raise e

# class OneAtaTimeTau(_BaseTauCorrector):
#     """A tau corrector that computes a tau correction for each minibatch,
#     and cycles through the corrections one at a time.
#     """

#     def __init__(self, loss_fn):
#         super().__init__(loss_fn)
#         self.tau_corrections = None

#     def compute_tau(self, fine_level, coarse_level, dataloader, operators, verbose=False):
#         self.tau_corrections = []
#         for batch_idx, mini_batch_data in enumerate(dataloader):
#             fine_rhs = None
#             if fine_level.corrector.tau_corrections:
#                 fine_rhs = fine_level.corrector.tau_corrections[batch_idx]
#             curr_rhsW, curr_rhsB = self.get_tau_for_data(fine_level, coarse_level,
#                                                          (mini_batch_data,), operators,
#                                                          self.loss_fn, fine_rhs)
#             self.tau_corrections.append((curr_rhsW, curr_rhsB))

#     def correct(self, model, loss, batch_idx, num_batches, verbose=False):
#         if self.tau_corrections is not None:
#             rhsW, rhsB = self.tau_corrections[batch_idx]
#             try:
#                 for layer_id in range(len(model.layers)):
#                     loss -= torch.sum(torch.mul(model.layers[layer_id].weight, rhsW[layer_id]))
#                     loss -= torch.sum(torch.mul(model.layers[layer_id].bias, rhsB[layer_id]))
#             except Exception as e:
#                 print("Exception in tau_corrector.py:OneAtaTimeTau.correct.", file=sys.stderr)
#                 raise e
            
