"""
Holds Tau correction definitions
"""
from abc import ABC, abstractmethod
import torch
import MTNN.utils.logger as log
import MTNN.utils.printer as printer
import MTNN.core.multigrid.operators.interpolator as interp

import sys

log = log.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['BasicTau']

###################################################################
# Interface
####################################################################
class _BaseTauCorrector(ABC):
    """Overwrite this"""
    def __init__(self, loss_fn):
        """
        Attributes:
            loss_fn: <torch.nn.modules.loss> Loss function
            rhs_W: residual weights
            rhs_B: residual biases
        """
        self.loss_fn = loss_fn
        self.rhs_W = None
        self.rhs_B = None

    @abstractmethod
    def compute_tau(self, fine_level, coarse_level, dataloader, operators, **kwargs):
        """Computes residual tau of the coarse-level"""
        raise NotImplementedError

    @abstractmethod
    def correct(self, model, loss, num_batches, **kwargs):
        """Returns corrected loss """
        raise NotImplementedError

    def get_tau_for_data(self, fine_level, coarse_level, dataloader, operators,
                         loss_fn, verbose=False):
        """
        Compute the coarse-level residual tau correction. Returns
        Args:
             fine_level:  <MTNN.core.multigrid.scheme> Level
             coarse_level: <MTNN.core.multigrid.scheme> Level
             dataloader: <torch.utils.data.Dataloade>
             operators: <MTNN.utils.datatypes> namedtuple
             verbose: <bool> Prints statements to stdout
        Returns:
            None
        """
        # ==============================
        #  coarse_level rhs
        # ==============================
        assert fine_level.id < coarse_level.id
        num_fine_layers = len(fine_level.net.layers)

        # Get the residual
        # get the gradients on the fine level and coarse_level
        fine_level_grad = fine_level.net.getGrad(dataloader, loss_fn)
        coarse_level_grad = coarse_level.net.getGrad(dataloader, loss_fn)

        # coarse level: grad_{W,B} = R * [f^h - A^{h}(u)] + A^{2h}(R*u)
        rhs_W_array = []
        rhs_B_array = []
        for layer_id in range(len(fine_level.net.layers)):
            dW_f = fine_level_grad.weight_grad[layer_id]
            dB_f = fine_level_grad.bias_grad[layer_id]
            # f^h - A^h(u^h)
            if fine_level.id > 0:
                rhsW = fine_level.corrector.rhs_W[layer_id] - dW_f
                rhsB = fine_level.corrector.rhs_B[layer_id] - dB_f
            else:  # first level
                rhsW = -dW_f
                rhsB = -dB_f

            rhs_W_array.append(rhsW)
            rhs_B_array.append(rhsB)

        # R * [f^h - A^h(u^h)]
        coarse_level_rhsW, coarse_level_rhsB = interp.transfer(
            rhs_W_array, rhs_B_array, operators.R_for_grad_op, operators.P_for_grad_op)

        # R * [f^h - A^h(u^h)] + A^{2h}(R*u^h)
        for layer_id in range(len(fine_level.net.layers)):
            coarse_level_rhsW[layer_id] += coarse_level_grad.weight_grad[layer_id]
            coarse_level_rhsB[layer_id] += coarse_level_grad.bias_grad[layer_id]

        return coarse_level_rhsW, coarse_level_rhsB    

###################################################################
# Implementation
####################################################################
class BasicTau(_BaseTauCorrector):
    def __init__(self, loss_fn):
        super().__init__(loss_fn)

    def compute_tau(self, fine_level, coarse_level, dataloader, operators, verbose=False):
        (rhsW, rhsB) = self.get_tau_for_data(fine_level, coarse_level, dataloader,
                                             operators, self.loss_fn, verbose)
        coarse_level.corrector.rhs_W = rhsW
        coarse_level.corrector.rhs_B = rhsB
        
    def correct(self, model, loss, batch_idx, num_batches, verbose=False):
        if self.rhs_W and self.rhs_B is not None:
            try:
                for layer_id in range(len(model.layers)):
                    loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].weight, self.rhs_W[layer_id]))
                    loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].bias, self.rhs_B[layer_id]))

                if verbose:
                    printer.print_tau(self, loss.item(), msg="\tApplying Tau Correction: ")
            except Exception as e:
                print("Exception in tau_corrector.py:BasicTau.correct.", file=sys.stderr)
                raise e

class OneAtaTimeTau(_BaseTauCorrector):
    """A tau corrector that computes a tau correction for each minibatch,
    and cycles through the corrections one at a time.
    """

    def __init__(self, loss_fn):
        super().__init__(loss_fn)
        self.tau_corrections = None

    def compute_tau(self, fine_level, coarse_level, dataloader, operators, verbose=False):
        self.tau_corrections = []
        for batch_idx, mini_batch_data in enumerate(dataloader):
            curr_rhsW, curr_rhsB = self.get_tau_for_data(fine_level, coarse_level,
                                                         (mini_batch_data,), operators,
                                                    self.loss_fn, verbose)
            self.tau_corrections.append((curr_rhsW, curr_rhsB))

    def correct(self, model, loss, batch_idx, num_batches, verbose=False):
        if self.tau_corrections is not None:
            rhsW, rhsB = self.tau_corrections[batch_idx]
            try:
                for layer_id in range(len(model.layers)):
                    loss -= torch.sum(torch.mul(model.layers[layer_id].weight, rhsW[layer_id]))
                    loss -= torch.sum(torch.mul(model.layers[layer_id].bias, rhsB[layer_id]))
            except Exception as e:
                print("Exception in tau_corrector.py:OneAtaTimeTau.correct.", file=sys.stderr)
                raise e
            
