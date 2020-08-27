"""
Holds Tau correction definitions
"""
from abc import ABC, abstractmethod
import torch
import MTNN.utils.logger as log
import MTNN.utils.printer as printer

log = log.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['BasicTau']

###################################################################
# Interface
####################################################################
class _BaseTauCorrector(ABC):
    """Overwrite this"""
    def __init__(self):
        """
        Attributes:
            rhs_W: residual weights
            rhs_B: residual biases
        """
        self.rhs_W = None
        self.rhs_B = None

    @abstractmethod
    def compute_tau(self, **kwargs):
        """Computes residual tau of the coarse-level"""
        raise NotImplementedError

    @abstractmethod
    def correct(self, **kwargs):
        """Returns corrected loss """
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class BasicTau(_BaseTauCorrector):
    def __init__(self):
        super().__init__()

    def compute_tau(self, fine_level, coarse_level, dataloader, operators, verbose=False):
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
        num_fine_layers = len(fine_level.net.layers)

        # Get the residual
        # get the gradient on the fine level
        fine_level_grad = fine_level.net.getGrad(dataloader, fine_level.loss_fn)
        log.debug(f"Restriction.Fine Level after get grad{fine_level_grad =}")

        # get the gradient on the coarse level
        coarse_level_grad = coarse_level.net.getGrad(dataloader, coarse_level.loss_fn)
        log.debug(f"Restriction.Coarse Level after get grad {coarse_level_grad=}")

        # coarse level: grad_{W,B} = R * [f^h - A^{h}(u)] + A^{2h}(R*u)
        coarse_level_rhsW = []
        coarse_level_rhsB = []

        # Compute the Tau correction
        for layer_id in range(len(fine_level.net.layers)):

            dW_f = fine_level_grad.weight_grad[layer_id]
            dB_f = fine_level_grad.bias_grad[layer_id]
            dW_c = coarse_level_grad.weight_grad[layer_id]
            dB_c = coarse_level_grad.bias_grad[layer_id]

            # f^h - A^h(u^h)
            if fine_level.id > 0:
                rhsW = fine_level.corrector.rhs_W[layer_id] - dW_f
                rhsB = fine_level.corrector.rhs_B[layer_id] - dB_f
            else:  # first level
                rhsW = -dW_f
                rhsB = -dB_f

            # R * [f^h - A^h(u^h)]
            if layer_id < num_fine_layers - 1:
                if layer_id == 0:
                    rhsW = operators.R_op[layer_id] @ rhsW
                else:
                    rhsW = operators.R_op[layer_id] @ rhsW @ operators.P_op[layer_id - 1]
                rhsB = operators.R_op[layer_id] @ rhsB
            elif layer_id > 0:
                rhsW = rhsW @ operators.P_op[-1]

            # R * [f^h - A^{h}(u^h)] + A^{2h}(R*u^h)
            rhsW += dW_c
            rhsB += dB_c

            coarse_level_rhsW.append(rhsW)
            coarse_level_rhsB.append(rhsB)

        if verbose:
            log.info(f"Restriction.Tau {rhsW = } {rhsB =}")

        coarse_level.corrector.rhs_W = coarse_level_rhsW
        coarse_level.corrector.rhs_B = coarse_level_rhsB

    def correct(self, model, num_batches, loss, verbose=False):
        if self.rhs_W and self.rhs_B is not None:
            try:
                for layer_id in range(len(model.layers)):
                    loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].weight, self.rhs_W[layer_id]))
                    loss -= (1.0 / num_batches) * torch.sum(torch.mul(model.layers[layer_id].bias, self.rhs_B[layer_id]))

                if verbose:
                    printer.print_tau(self, loss, msg="\t")
            except Exception as e:
                raise e
