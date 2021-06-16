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

                
                WB = torch.cat((currW, currB), dim=1)
                nr = torch.norm(WB, p=2, dim=1, keepdim=True)
                
                R_layer = torch.zeros([nC, nF]).to(op_device)
                for i in range(nF):
                    R_layer[F2C_layer[i], i] = 1

                d = (R_layer * torch.reshape(nr, (1, -1)) @ torch.transpose(R_layer, 0, 1)).diagonal().reshape(-1, 1)
            
                # P = diag(nr) * R^T * diag(1./d)
                restriction_operators.append(R_layer)

                # Construct prolongation operators for each layer
                # TODO - Decouple?
                P_layer = (torch.transpose(R_layer, 0, 1) * nr) / d.reshape(1, -1)
                prolongation_operators.append(P_layer)

                R_for_grad_layer = torch.transpose(P_layer, 0, 1)
                restriction_for_grad_operators.append(R_for_grad_layer)

                P_for_grad_layer = torch.transpose(R_layer, 0, 1)
                prolongation_for_grad_operators.append(P_for_grad_layer)

        return TransferOps(restriction_operators, prolongation_operators,
                           restriction_for_grad_operators, prolongation_for_grad_operators)
