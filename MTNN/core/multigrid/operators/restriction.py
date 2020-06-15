"""
Restriction Operators
"""
import torch
import numpy as np
from abc import ABC, abstractmethod

# local
import MTNN.core.multigrid.operators.coarsener as coarsener
import MTNN.utils.logger as log

log = log.get_logger(__name__, write_to_file =True)


###################################################################
# Interface
####################################################################
class _BaseRestriction(ABC):
    """Overwrite this"""
    @abstractmethod
    def apply(self, **kawrgs):
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class PairwiseAggregationRestriction(_BaseRestriction):
    """
    Pairwise Aggregation-based Restriction Operator for Fully connected Networks
        * Uses Heavy Edge matching from similarity matrix of source_model's Weight and Bias

    Returns Coarsened Model/grid
    """
    def __init__(self):
        # A list of restriction operators(matrices) to use
        # for each fine_level network's hidden layer

        self.prolongation_operators = []
        self.restriction_operators = []
        self.coarsener = coarsener._HEMCoarsener()

    def apply(self, fine_level, coarse_level, verbose=False):
        self._setup(fine_level, coarse_level)


        # Update coarse level's layers by applying self.restriction_operators
        # TODO: Fill with agg_interpolator.restrict
        num_layers = len(fine_level.net.layers)
        coarse_level.Winit_array = []
        coarse_level.Binit_array = []
        # ==============================
        #  coarse_level weight and bias
        # ==============================
        print("Applying Restriction")
        for layer_id in range(num_layers):
            W_f = fine_level.net.layers[layer_id].weight.detach().numpy()
            B_f = fine_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1)

            if layer_id < num_layers - 1:
                if layer_id == 0:
                    print("Agg: restrict:First network layer", self.restriction_operators[layer_id])
                    log.info(f"Restriction operator{np.shape(self.restriction_operators[layer_id])} Fine-level weight{np.shape(W_f)}")
                    W_c = self.restriction_operators[layer_id] @ W_f
                else:
                    W_c = self.restriction_operators[layer_id] @ W_f @ self.prolongation_operators[layer_id - 1]
                B_c = self.restriction_operators[layer_id] @ B_f
            elif layer_id > 0:
                W_c = W_f @ self.prolongation_operators[-1]
                B_c = np.copy(B_f)

            # save the initial W_c and B_c
            # do we use the initial W_c and B_c?
            coarse_level.Winit_array.append(W_c)
            coarse_level.Binit_array.append(B_c)
            #
            assert coarse_level.net.layers[layer_id].weight.detach().numpy().shape == W_c.shape
            assert coarse_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1).shape == B_c.shape
            with torch.no_grad():
                np.copyto(coarse_level.net.layers[layer_id].weight.detach().numpy(), W_c)
                np.copyto(coarse_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1), B_c)

        coarse_level.net.zero_grad()

        """
        # ==============================
        #  coarse_level rhs
        # ==============================
        # get the gradient on the fine level
        fine_level.net.getgrad(dataset, fine_level.obj_func)
        # get the gradient on the coarse level
        coarse_level.net.getgrad(dataset, fine_level.obj_func)
        # coarse level: grad_{W,B} = R * [f^h - A^{h}(u)] + A^{2h}(R*u)
        coarse_level.rhs_W_array = []
        coarse_level.rhs_B_array = []
        for layer_id in range(num_layers):
            dW_f = np.copy(fine_level.net.layers[layer_id].weight.grad.detach().numpy())
            dB_f = np.copy(fine_level.net.layers[layer_id].bias.grad.detach().numpy().reshape(-1, 1))
            dW_c = np.copy(coarse_level.net.layers[layer_id].weight.grad.detach().numpy())
            dB_c = np.copy(coarse_level.net.layers[layer_id].bias.grad.detach().numpy().reshape(-1, 1))

            # f^h - A^h(u^h)
            if fine_level.level_id > 0:
                rhsW = fine_level.rhs_W_array[layer_id] - dW_f
                rhsB = fine_level.rhs_B_array[layer_id] - dB_f
            else:
                rhsW = -dW_f
                rhsB = -dB_f

            # R * [f^h - A^h(u^h)]
            if layer_id < num_layers - 1:
                if layer_id == 0:
                    rhsW = self.R_array[layer_id] @ rhsW
                else:
                    rhsW = self.R_array[layer_id] @ rhsW @ self.P_array[layer_id - 1]
                rhsB = self.R_array[layer_id] @ rhsB
            elif layer_id > 0:
                rhsW = rhsW @ self.P_array[-1]

            # R * [f^h - A^{h}(u^h)] + A^{2h}(R*u^h)
            rhsW += dW_c
            rhsB += dB_c

            coarse_level.rhs_W_array.append(rhsW)
            coarse_level.rhs_B_array.append(rhsB)
        """

    def _setup(self, fine_level, coarse_level):
        """
        Construct the restriction operators based on Heavy Edge Matching
        Returns:
            None
        """


        #log.info(f"Restriction:Setup {fine_level.net.__class__}")
        #log.info(f"Restrictions: settings {fine_level.net.activation}")


        # Instantiate the coarse-level net with the coarsener dimensions
        self.coarsener.coarsen(fine_level.net)
        log.info(f"Coarse Level Net dimensions: {self.coarsener.coarseLevelDim}")

        coarse_level.net = fine_level.net.__class__(self.coarsener.coarseLevelDim,
                                                    fine_level.net.activation,
                                                    fine_level.net.output)
        log.info(f"Coarse level after: {coarse_level.net}")

        # Create the restriction operator per hidden layer (except for the last layer)
        # from the coarsener

        num_layers = len(fine_level.net.layers)
        for layer_id in range(num_layers - 1):
            log.info(f"Build Restriction Operators for layer {layer_id}")
            original_W = np.copy(fine_level.net.layers[layer_id].weight.detach().numpy())
            original_b = np.copy(fine_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1))

            WB = np.concatenate([original_W, original_b], axis = 1)
            nr = np.linalg.norm(WB, ord = 2, axis = 1, keepdims = True)

            F2C_l = self.coarsener.Fine2CoursePerLayer[layer_id]

            nF = fine_level.net.layers[layer_id].out_features
            nC = coarse_level.net.layers[layer_id].out_features
            R_l = np.zeros([nC, nF])

            # Construct restriction operator
            for i in range(nF):
                #print(f"Constructing R: {i}, {F2C_l[i]}")
                R_l[F2C_l[i], i] = 1
            # d = diag(R*B*R^T)
            d = (R_l * nr.reshape(1, -1) @ np.transpose(R_l)).diagonal().reshape(-1, 1)
            # P = diag(nr) * R^T * diag(1./d)

            # Construct prolongation operator
            #TODO: Decouple. Move to Prolongation.py?
            P_l = np.transpose(R_l) * nr / d.reshape(1, -1)

            log.info(f"\n R {R_l} \n P {P_l}")
            # pdb.set_trace()
            self.restriction_operators.append(R_l)
            self.prolongation_operators.append(P_l)





