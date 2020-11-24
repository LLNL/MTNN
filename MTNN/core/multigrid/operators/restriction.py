"""
Restriction Operators
"""
import torch
import torch.nn as nn
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

    def apply(self, fine_level, coarse_level, dataloader, verbose=False) -> None:
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
        if fine_level.interpolation_data is None or self.resetup == 100:
            fine_level.interpolation_data = interp.PairwiseAggCoarsener().setup(fine_level, coarse_level)
            self.resetup = 0
        else:
            self.resetup += 1
            
        R_op, P_op = fine_level.interpolation_data.R_op, fine_level.interpolation_data.P_op
        R_for_grad_op = fine_level.interpolation_data.R_for_grad_op
        P_for_grad_op = fine_level.interpolation_data.P_for_grad_op

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
                coarse_level.net.layers[layer_id].weight.copy_(W_c.clone()) # = nn.Parameter(W_c.clone())
                coarse_level.net.layers[layer_id].bias.copy_(B_c.clone().reshape(-1)) # = nn.Parameter(B_c.clone().reshape(-1))
                # W_c = coarse_level.net.layers[layer_id].weight.detach().clone()
                # B_c = coarse_level.net.layers[layer_id].bias.detach().clone().reshape(-1, 1)

        coarse_level.net.zero_grad()


        # Restrict tau
        ops = operators(R_op, P_op, R_for_grad_op, P_for_grad_op)
        coarse_level.corrector.compute_tau(fine_level, coarse_level, dataloader, ops, verbose)


        # Restrict momentum
        # Momentum has the same form as the network parameters, so use the same algorithm
        assert(len(fine_level.presmoother.optimizer.param_groups) == 1)
        fine_optimizer = fine_level.presmoother.optimizer
        coarse_level.Wmomentum_init = []
        coarse_level.Bmomentum_init = []
        coarse_level.presmoother.momentum_data = []
        for i in range(0, len(fine_optimizer.param_groups[0]['params']), 2):
            mW_f = fine_optimizer.state[fine_optimizer.param_groups[0]['params'][i]]['momentum_buffer']
            mB_f = fine_optimizer.state[fine_optimizer.param_groups[0]['params'][i+1]]['momentum_buffer'].reshape(-1, 1)

            layer_id = int(i / 2)
            if layer_id < num_fine_layers - 1:
                if layer_id == 0:
                    mW_c = R_op[layer_id] @ mW_f
                else:
                    mW_c = R_op[layer_id] @ mW_f @ P_op[layer_id - 1]
                mB_c = R_op[layer_id] @ mB_f
            elif layer_id > 0:
                mW_c = mW_f @ P_op[-1]
                mB_c = mB_f.clone()

            # save the initial W_c and B_c
            # NOTE: Wmomentum_init and the Bmomentum_init only used in prolongation
            coarse_level.Wmomentum_init.append(mW_c)
            coarse_level.Bmomentum_init.append(mB_c)

            # The coarse optimizer may not exist yet, so store the
            # momentum data to be inserted in right before smoothing.
            with torch.no_grad():
                coarse_level.presmoother.momentum_data.append(mW_c.clone())
                coarse_level.presmoother.momentum_data.append(mB_c.clone().reshape(-1))


