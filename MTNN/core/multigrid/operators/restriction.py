"""
Restriction Operators
"""
import torch
import numpy as np
from abc import ABC, abstractmethod

# local
import MTNN.utils.logger as log
import MTNN.core.components.multigrid.operators.interpolator as interp

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

    Returns level.net - coarsened model (grid)
    """

    """"
    def __init__(self):
        # A list of restriction operators(matrices) to use
        # for each fine_level network's hidden layer

        # Operators corresponds to each layer
        #self.prolongation_operators = []
        #self.restriction_operators = []
       #self.coarsener = coarsener._HEMCoarsener()
    
    """

    def apply(self, fine_level, coarse_level, dataloader, verbose=False):
        """
        Apply Restriction on the fine_level to return a restricted coarse-level
        and compute the rhs with tau correction calculated per minibatch
        """

        # setup
        fine_level.interpolation_data = interp.PairwiseAggregationCoarsener().setup(fine_level, coarse_level)
        assert fine_level.interpolation_data is not None

        R_op, P_op = fine_level.interpolation_data.R_op, fine_level.interpolation_data.P_op

        # Update coarse level's layers by applying self.restriction_operators
        # TODO: Fill with agg_interpolator.restrict
        num_fine_layers = len(fine_level.net.layers)
        coarse_level.Winit_array = []
        coarse_level.Binit_array = []


        # ==============================
        #  coarse_level weight and bias
        # ==============================
        log.info("APPLYING RESTRICTION")

        for layer_id in range(num_fine_layers):
            W_f = fine_level.net.layers[layer_id].weight.detach().numpy()
            B_f = fine_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1)

            if layer_id < num_fine_layers - 1:
                if layer_id == 0:
                    #log.info(f"Agg: restrict:First network layer {self.restriction_operators[layer_id]}")
                    log.debug(f"Layer {layer_id} Restriction operator{np.shape(fine_level.interpolation_data.R_op[layer_id])} Fine-level weight{np.shape(W_f)}")
                    W_c = R_op[layer_id] @ W_f

                else:
                    log.debug( f"Layer {layer_id} Restriction operator{np.shape(fine_level.interpolation_data.R_op[layer_id])} Fine-level weight{np.shape(W_f)}")
                    W_c = R_op[layer_id] @ W_f @ P_op[layer_id - 1]
                B_c = R_op[layer_id] @ B_f
            elif layer_id > 0:

                W_c = W_f @ P_op[-1]
                B_c = np.copy(B_f)


            # save the initial W_c and B_c
            # NOTE: Winit and the Binit only used in prolongation
            coarse_level.Winit_array.append(W_c)
            coarse_level.Binit_array.append(B_c)

            assert coarse_level.net.layers[layer_id].weight.detach().numpy().shape == W_c.shape
            assert coarse_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1).shape == B_c.shape
            # Copy to coarse_level net

            with torch.no_grad():
                np.copyto(coarse_level.net.layers[layer_id].weight.detach().numpy(), W_c)
                np.copyto(coarse_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1), B_c)

        log.debug(f"restriction:apply {coarse_level.net.layers=}")
        coarse_level.net.zero_grad()

    def get_tau(self, fine_level, coarse_level, dataloader):
        # TODO: Separate TAU
        # ==============================
        #  coarse_level rhs
        # ==============================
      
        # get the gradient on the fine level
        fine_level.net._get_grad(dataloader, fine_level.loss_fn, fine_level.rhs_Warray, fine_level.rhs_Barray)
        # get the gradient on the coarse level
        coarse_level.net._get_grad(dataloader, fine_level.loss_fn, coarse_level.rhs_Warray, coarse_level.rhs_Barray)
        # coarse level: grad_{W,B} = R * [f^h - A^{h}(u)] + A^{2h}(R*u)
        coarse_level.rhs_W_array = []
        coarse_level.rhs_B_array = []

        # Compute the Tau
        for layer_id in range(len(fine_level.net.layers)):
            dW_f = np.copy(fine_level.net.layers[layer_id].weight.grad.detach().numpy())
            dB_f = np.copy(fine_level.net.layers[layer_id].bias.grad.detach().numpy().reshape(-1, 1))
            dW_c = np.copy(coarse_level.net.layers[layer_id].weight.grad.detach().numpy())
            dB_c = np.copy(coarse_level.net.layers[layer_id].bias.grad.detach().numpy().reshape(-1, 1))

            # f^h - A^h(u^h)
            if fine_level.id > 0:
                rhsW = fine_level.rhs_W_array[layer_id] - dW_f
                rhsB = fine_level.rhs_B_array[layer_id] - dB_f
            else:
                rhsW = -dW_f
                rhsB = -dB_f

            # R * [f^h - A^h(u^h)]
            if layer_id < num_fine_layers - 1:
                if layer_id == 0:
                    rhsW = R_op[layer_id] @ rhsW
                else:
                    rhsW = R_op[layer_id] @ rhsW @ P_op[layer_id - 1]
                rhsB = R_op[layer_id] @ rhsB
            elif layer_id > 0:
                rhsW = rhsW @ P_op[-1]

            # R * [f^h - A^{h}(u^h)] + A^{2h}(R*u^h)
            rhsW += dW_c
            rhsB += dB_c

            coarse_level.rhs_W_array.append(rhsW)
            coarse_level.rhs_B_array.append(rhsB)



