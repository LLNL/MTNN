# standard 
import collections as col
# torch 
import torch
# local
import MTNN.core.multigrid.scheme as mg
import MTNN.utils.logger as log
import MTNN.utils.datatypes as mgdata
from MTNN.core.multigrid.operators.SecondOrderRestriction import ParamLibrary, CoarseMapping, TransferOps

log = log.get_logger(__name__, write_to_file = True)

class PairwiseOpsBuilder:
    """
    PairwiseOpsBuilder is an object which is used to construct the actual 
    restriction and prolongation operators based on pairwise matching.
    """
    def __init__(self, restriction_weighting_power = 0.0, weighted_projection = True):
        """Constructor.

        Either sums or adds fine parameters into coarse ones for
        restriction. The prolongation operator is constructed to be a
        projection with the restriction, ie P is chosen so that PR is
        a projection. More specifically, we have
        PR = B R_0^T (R_0^T B R_0)^{-1} R_0,
        where B is a diagonal matrix of weights given by neuron vector norms, 
        and R_0 is the simple addition restriction operator.
        Depending on choices, we either set
        R = R_0
        or
        R = (R_0^T B R_0)^{-1} R_0,
        and then choose P accordingly.

        Inputs:
        sum_on_restriction (bool) - If true, restriction operator sums fine 
            parameters into coarse. If false, restriction operator performs 
            a weighted average.
        weighted_projection (bool) - If true, compute restriction/prolongation 
            as a projection weighted by vector norms. This choice ensures the 
            projection exactly reconstructing two fine neurons if they point in 
            exactly the same direction.

        """
        self.rwp = restriction_weighting_power
        self.weighted_projection = weighted_projection
    
    def __call__(self, param_library, coarse_mapping, op_device):
        """
        Inputs:
        param_library <ParamLibrary> - The parameters associated with the network.
        coarse_mapping <CoarseMapping> - The mapping from fine to coarse channels.
        op_device <torch.device> - The device on which the ops matrices should reside.
        """
        
        # Each index of Prolongation/Restriction operators corresponds to layer to which it is to be applied
        restriction_operators = []
        prolongation_operators = []
        restriction_for_grad_operators = []
        prolongation_for_grad_operators = []

        fine2coarse, num_coarse_array = coarse_mapping
        with torch.no_grad():
            W_f_array, B_f_array = param_library
            for layer_id in range(len(W_f_array)-1):
                F2C_layer = fine2coarse[layer_id]
                nF = len(F2C_layer)
                nC = num_coarse_array[layer_id]
                assert(W_f_array[layer_id].shape[-2] == nF)
                
                currW = W_f_array[layer_id].transpose(0, -2).flatten(1)
                currB = B_f_array[layer_id]


                if self.weighted_projection:
                    WB = torch.cat((currW, currB), dim=1)
                    nr = torch.norm(WB, p=2, dim=1, keepdim=True)
                else:
                    nr = torch.ones(currW.shape[0], 1)
                
                R_layer = torch.zeros([nC, nF]).to(op_device)
                for i in range(nF):
                    R_layer[F2C_layer[i], i] = 1

                
                d = (R_layer * torch.reshape(nr, (1, -1)) @ torch.transpose(R_layer, 0, 1)).diagonal()#.reshape(-1, 1)
                # R = R * diag(1./d)^rwp
                R_layer = R_layer / torch.pow(d, self.rwp).reshape(-1, 1)
                # P = diag(nr) * R^T * diag(1./d)^(1-rwp)
                P_layer = (torch.transpose(R_layer, 0, 1) * nr) / torch.pow(d, 1.0 - self.rwp).reshape(1, -1)
                
                # if self.sum_on_restriction:
                #     # Construct prolongation operators for each layer
                #     # TODO - Decouple?

                #     P_layer = (torch.transpose(R_layer, 0, 1) * nr) / d.reshape(1, -1)
                # else:
                #     R_layer = R_layer / d.reshape(-1, 1)
                #     P_layer = torch.transpose(R_layer, 0, 1) * nr

                restriction_operators.append(R_layer)

                prolongation_operators.append(P_layer)

                R_for_grad_layer = torch.transpose(P_layer, 0, 1)
                restriction_for_grad_operators.append(R_for_grad_layer)

                P_for_grad_layer = torch.transpose(R_layer, 0, 1)
                prolongation_for_grad_operators.append(P_for_grad_layer)

                # if self.adjust_bias:
                #     bias_restriction_operators(adjustments * R_layer)
                #     bias_prolongation_operators(P_layer / adjustments)
                    
        return TransferOps(restriction_operators, prolongation_operators,
                           restriction_for_grad_operators, prolongation_for_grad_operators)

