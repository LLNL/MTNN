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
        l2reg_left_vecs = fine_level.interpolation_data.l2reg_left_vecs
        l2reg_right_vecs = fine_level.interpolation_data.l2reg_right_vecs

        # Update coarse level's layers by applying self.restriction_operators
        # TODO: Fill with agg_interpolator.restrict
        num_fine_layers = len(fine_level.net.layers)
        coarse_level.Winit = []
        coarse_level.Binit = []


        # ==============================
        #  coarse_level weight and bias
        # ==============================
        W_f_array = [fine_level.net.layers[layer_id].weight.detach() for layer_id in range(num_fine_layers)]
        B_f_array = [fine_level.net.layers[layer_id].bias.detach().reshape(-1, 1) for layer_id in range(num_fine_layers)]
        W_c_array, B_c_array = interp.transfer(W_f_array, B_f_array, R_op, P_op)
        with torch.no_grad():
            for layer_id in range(num_fine_layers):
                coarse_level.net.layers[layer_id].weight.copy_(W_c_array[layer_id])
                coarse_level.net.layers[layer_id].bias.copy_(B_c_array[layer_id].reshape(-1))
        coarse_level.Winit = W_c_array
        coarse_level.Binit = B_c_array
            
        coarse_level.net.zero_grad()


        # Restrict tau
        coarse_level.corrector.compute_tau(fine_level, coarse_level, dataloader,
                                           fine_level.interpolation_data, verbose)

        # ===============================================================================
        # Restrict momentum
        # Momentum has the same form as the network parameters, so use the same algorithm
        # ===============================================================================
        fine_optimizer = fine_level.presmoother.optimizer
        get_p = lambda ind : fine_optimizer.state[fine_optimizer.param_groups[0]['params'][ind]]['momentum_buffer']
        mW_f_array, mB_f_array = zip(*[(get_p(2*i), get_p(2*i+1).reshape(-1, 1)) for i in
                                       range(int(len(fine_optimizer.param_groups[0]['params']) / 2))])
        mW_c_array, mB_c_array = interp.transfer(mW_f_array, mB_f_array, R_op, P_op)
        
        assert(len(mW_c_array) == len(mB_c_array))
        coarse_level.presmoother.momentum_data = []
        with torch.no_grad():
            for i in range(len(mW_c_array)):
                coarse_level.presmoother.momentum_data.append(mW_c_array[i].clone())
                coarse_level.presmoother.momentum_data.append(mB_c_array[i].clone().reshape(-1))
        coarse_level.Wmomentum_init = mW_c_array
        coarse_level.Bmomentum_init = mB_c_array
    


        # ===========================================
        # Compute l2 regularization correction vector
        # z = 2 R (I - P Pi) x_old
        # ===========================================
        # z1 = P Pi x_old, already have x_c = Pi x_old
        W_f_proj, B_f_proj = interp.transfer(W_c_array, B_c_array, P_op, R_op)
        # z2 = 2 (I - P Pi) x_old = 2*(x_old - z1)
        z2W = []
        z2B = []
        with torch.no_grad():
            for layer_id in range(num_fine_layers):
                z2W.append(2.0*(fine_level.net.layers[layer_id].weight.detach() - W_f_proj[layer_id]))
                z2B.append(2.0*(fine_level.net.layers[layer_id].bias.detach() - B_f_proj[layer_id].reshape(-1)))
        # z = R z2
        zW, zB = interp.transfer(z2W, z2B, R_for_grad_op, P_for_grad_op)
        coarse_level.l2_info = (zW, zB, l2reg_left_vecs, l2reg_right_vecs)        
        
