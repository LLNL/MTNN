"""
Restriction Operators
"""
import torch
import numpy as np
from abc import ABC, abstractmethod

# local
import MTNN.core.multigrid.scheme as mg
import MTNN.core.multigrid.operators.interpolator as interp
from MTNN.utils.datatypes import operators
import MTNN.utils.logger as log
import MTNN.utils.printer as printer

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['PairwiseAggRestriction']
###################################################################
# Interface
####################################################################
class _BaseRestriction(ABC):
    """Overwrite this"""

    @abstractmethod
    def apply(self, **kwargs):
        raise NotImplementedError

###################################################################
# Implementation
####################################################################
class PairwiseAggRestriction(_BaseRestriction):
    """
    Pairwise Aggregation-based Restriction Operator for Fully connected Networks
        * Uses Heavy Edge matching from similarity matrix of source_model's Weight and Bias

    Restricts fine level model and its residual to create a coarsened model with rhs.
    """

    def __init__(self, matching_alg):
        self.matching_alg = matching_alg

    def apply(self, fine_level, coarse_level, dataloader, corrector, verbose=False) -> None:
        """
        Apply Restriction on the fine_level to return a restricted coarse-level net
        and compute the coarse-level's residual tau correction.
        Args:
            fine_level: <core.multigrid.scheme> Level
            coarse_level: <core.multigrid.scheme>  Level
            dataloader: <torch.utils.data.Dataloader> Pytorch dataloader
            corrector: <core.multigrid.operators.tau_corrector> _BaseTau_Corrector
            verbose: <bool>

        Returns:
            None
        """


        # setup
        # TODO: refactor setup
        assert fine_level.id < coarse_level.id
        fine_level.interpolation_data = interp.PairwiseAggCoarsener().setup(fine_level, coarse_level)
        assert fine_level.interpolation_data is not None

        R_op, P_op = fine_level.interpolation_data.R_op, fine_level.interpolation_data.P_op

        # Update coarse level's layers by applying self.restriction_operators
        # TODO: Fill with agg_interpolator.restrict
        num_fine_layers = len(fine_level.net.layers)
        coarse_level.Winit = []
        coarse_level.Binit = []


        # ==============================
        #  coarse_level weight and bias
        # ==============================
        for layer_id in range(num_fine_layers):
            W_f = fine_level.net.layers[layer_id].weight.detach()
            B_f = fine_level.net.layers[layer_id].bias.detach().reshape(-1, 1)

            if layer_id < num_fine_layers - 1:
                if layer_id == 0:
                    W_c = R_op[layer_id] @ W_f
                else:
                    W_c = R_op[layer_id] @ W_f @ P_op[layer_id - 1]
                B_c = R_op[layer_id] @ B_f
            elif layer_id > 0:
                W_c = W_f @ P_op[-1]
                B_c = B_f.clone()


            # save the initial W_c and B_c
            # NOTE: Winit and the Binit only used in prolongation
            coarse_level.Winit.append(W_c)
            coarse_level.Binit.append(B_c)

            assert coarse_level.net.layers[layer_id].weight.detach().clone().shape == W_c.shape
            assert coarse_level.net.layers[layer_id].bias.detach().clone().reshape(-1, 1).shape == B_c.shape
            # Copy to coarse_level net

            with torch.no_grad():
                W_c = coarse_level.net.layers[layer_id].weight.detach().clone()
                B_c = coarse_level.net.layers[layer_id].bias.detach().clone().reshape(-1, 1)


        coarse_level.net.zero_grad()

        ops = operators(R_op, P_op)
        corrector.compute_tau(fine_level, coarse_level, dataloader, ops, verbose)


