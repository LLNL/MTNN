# standard 
import collections as col
# torch 
import torch
# local
import MTNN.core.multigrid.scheme as mg
import MTNN.utils.logger as log
from MTNN.utils.datatypes import ParamVector, CoarseMapping, TransferOps

log = log.get_logger(__name__, write_to_file = True)

# Xc = R * Xf
def PreMultR(match, Xf):
    nf = match.size()[0]
    pair2 = match[match >= torch.arange(0, nf, device=match.device)]
    pair1 = match[pair2]
    Xc = Xf[pair1, :] + Xf[pair2, :]
    # singleton
    Xc[pair1 == pair2, :] /= 2
    return Xc

class PairwiseOpsBuilder:
    """
    PairwiseOpsBuilder is an object which is used to construct the actual 
    restriction and prolongation operators based on pairwise matching.
    """
    def __init__(self, weighted_projection = True):
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
        weighted_projection (bool) - If true, compute restriction/prolongation 
            as a projection weighted by vector norms. This choice ensures the 
            projection exactly reconstructing two fine neurons if they point in 
            exactly the same direction.

        """
        self.weighted_projection = weighted_projection
    
    def __call__(self, param_library, coarse_mapping, op_device):
        """
        Inputs:
        param_library <ParamVector> - The parameters associated with the network.
        coarse_mapping <CoarseMapping> - The mapping from fine to coarse channels.
        op_device <torch.device> - The device on which the ops matrices should reside.
        """
        
        # Each index of Prolongation/Restriction operators corresponds to layer to which it is to be applied
        restriction_operators = []
        prolongation_operators = []
        restriction_for_grad_operators = []
        prolongation_for_grad_operators = []

        fine2coarse, num_coarse_array = coarse_mapping.fine2coarse_map, coarse_mapping.num_coarse_channels
        with torch.no_grad():
            W_f_array, B_f_array = param_library.weights, param_library.biases
            for layer_id in range(len(W_f_array)-1):
                F2C_layer = fine2coarse[layer_id]
                nF = len(F2C_layer)
                nC = num_coarse_array[layer_id]
                assert(W_f_array[layer_id].shape[-2] == nF)
                
                currW = W_f_array[layer_id].transpose(0, -2).flatten(1)
                currB = B_f_array[layer_id]


                if self.weighted_projection:
                    WB = torch.cat((currW, currB), dim=1)
                    nr = torch.norm(WB, p=2, dim=1, keepdim=True).to(op_device)
                else:
                    nr = torch.ones(currW.shape[0], 1).to(op_device)
                
                R_layer = torch.zeros([nC, nF]).to(op_device)
                for i in range(nF):
                    R_layer[F2C_layer[i], i] = 1

                d = (R_layer * torch.reshape(nr, (1, -1)) @ torch.transpose(R_layer, 0, 1)).diagonal().reshape(-1, 1)

                # P = diag(nr) * R^T * diag(1./d)
                P_layer = (torch.transpose(R_layer, 0, 1) * nr) / d.reshape(1, -1)
                
                restriction_operators.append(R_layer)
                prolongation_operators.append(P_layer)

                R_for_grad_layer = torch.transpose(P_layer, 0, 1)
                restriction_for_grad_operators.append(R_for_grad_layer)

                P_for_grad_layer = torch.transpose(R_layer, 0, 1)
                prolongation_for_grad_operators.append(P_for_grad_layer)

        return TransferOps(restriction_operators, prolongation_operators), TransferOps(restriction_for_grad_operators, prolongation_for_grad_operators)






class R_Multiplier:
    """Represents an R matrix in the layer-by-layer transfer operators.

    If W is a weight matrix, R W_f P restricts the outputs to a set of
    coarse neurons. During prolongation, P W_c R prolongs the coarse
    neurons back to fine neurons.
    """
    
    def __init__(self, match):
        self.match = match

    def transpose(self):
        return R_Transpose_Multiplier(self.match)

    def __matmul__(self, Xf):
        """ Use this operator as left-multiplication, as used during restriction.

        Xc = R * Xf
        """
        nf = self.match.size()[0]
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xc = Xf[..., pair1, :] + Xf[..., pair2, :]
        # singleton
        Xc[..., pair1 == pair2, :] /= 2
        return Xc

    def __rmatmul__(self, Xc):
        """ Use this operator as right-multiplication, as used during prolongation.

        Xf = Xc * R
        """
        nf = self.match.size()[0]
        Xf = torch.empty((*Xc.size()[:-1], nf), device=Xc.device)
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xf[..., pair2] = Xc
        Xf[..., pair1] = Xc
        return Xf


class P_Multiplier:
    """Represents a P matrix in the layer-by-layer transfer operators.

    If W is a weight matrix, R W_f P restricts the outputs to a set of
    coarse neurons. During prolongation, P W_c R prolongs the coarse
    neurons back to fine neurons.
    """

    def __init__(self, match, d1, d2):
        self.match = match
        self.d1 = d1
        self.d2 = d2

    def transpose(self):
        return P_Transpose_Multiplier(self.match, self.d1, self.d2)

    def __matmul__(self, Xc):
        """ Use this operator as left-multiplication, as used during prolongation.

        Xf = P * Xc = D1 * R^{T} * D2 * Xc
        """
        nf = self.match.size()[0]
        Xf = torch.empty((*Xc.shape[:-2], nf, Xc.shape[-1]), device=Xc.device)
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xc = Xc * self.d2.reshape(-1, 1) # What does this line do with higher order tensors?
        Xf[..., pair2, :] = Xc
        Xf[..., pair1, :] = Xc
        Xf = Xf * self.d1.reshape(-1, 1) # What does this line do with higher order tensors?
        return Xf

    def __rmatmul__(self, Xf):
        """ Use this operator as right-multiplication, as used during restriction.

        Xc = Xf * P = Xf * D1 * R^{T} * D2
        """
        nf = self.match.size()[0]
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xf = Xf * self.d1.reshape(1, -1) # Computes Xf[...,i] * d1[i] for each i
        Xc = Xf[..., pair1] + Xf[..., pair2]
        # singleton
        Xc[..., pair1 == pair2] /= 2
        Xc = Xc * self.d2.reshape(1, -1) # Compute Xf[...,i] * d2[i] for each i
        return Xc



class R_Transpose_Multiplier:
    """Represents the tranpose of an R matrix in the layer-by-layer transfer operators.

    This is used to compute the tau correction; specifically, one computes P^T W R^T.
    """
    
    def __init__(self, match):
        self.match = match

    def __rmatmul__(self, Xf):
        """ Use this operator as right-multiplication, as used during restriction for tau.

        Xc = Xf * R^T
        """
        nf = self.match.size()[0]
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xc = Xf[..., pair1] + Xf[..., pair2]
        # singleton
        Xc[..., pair1 == pair2] /= 2
        return Xc
    
class P_Transpose_Multiplier:
    """Represents the transpose of a P matrix in the layer-by-layer transfer operators.

    This is used to the compute the tau correction; specifically, one computes P^T W R^T.
    """
    def __init__(self, match, d1, d2):
        self.match = match
        self.d1 = d1
        self.d2 = d2

    def __matmul__(self, Xf):
        """Use this operator as left-multiplication, as used during restriction for tau.

        Xc = P^T * Xf = D2 * R * D1 * Xf
        """
        nf = self.match.size()[0]
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xf = Xf * self.d1.reshape(-1, 1)
        Xc = Xf[..., pair1, :] + Xf[..., pair2, :]
        # singleton
        Xc[..., pair1 == pair2, :] /= 2
        Xc = Xc * self.d2.reshape(-1, 1)
        return Xc


class PairwiseOpsBuilder_MatrixFree:
    """
    PairwiseOpsBuilder is an object which is used to construct the actual 
    restriction and prolongation operators based on pairwise matching.
    """
    def __init__(self, weighted_projection = True):
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
        weighted_projection (bool) - If true, compute restriction/prolongation 
            as a projection weighted by vector norms. This choice ensures the 
            projection exactly reconstructing two fine neurons if they point in 
            exactly the same direction.

        """
        self.weighted_projection = weighted_projection
    
    def __call__(self, param_library, coarse_mapping, op_device):
        """
        Inputs:
        param_library <ParamVector> - The parameters associated with the network.
        coarse_mapping <CoarseMapping> - The mapping from fine to coarse channels.
        op_device <torch.device> - The device on which the ops matrices should reside.
        """
        
        # Each index of Prolongation/Restriction operators corresponds to layer to which it is to be applied
        restriction_operators = []
        prolongation_operators = []
        restriction_for_grad_operators = []
        prolongation_for_grad_operators = []

        D2 = []

        fine2coarse, num_coarse_array = coarse_mapping.fine2coarse_map, coarse_mapping.num_coarse_channels
        match_per_layer = coarse_mapping.match_per_layer
        with torch.no_grad():
            W_f_array, B_f_array = param_library.weights, param_library.biases
            for layer_id in range(len(W_f_array)-1):
                nF = len(match_per_layer[layer_id])
                nC = num_coarse_array[layer_id]
                assert(W_f_array[layer_id].shape[-2] == nF)
                
                currW = W_f_array[layer_id].transpose(0, -2).flatten(1)
                currB = B_f_array[layer_id]


                if self.weighted_projection:
                    WB = torch.cat((currW, currB), dim=1)
                    nr = torch.norm(WB, p=2, dim=1, keepdim=True).to(op_device)
                else:
                    nr = torch.ones(currW.shape[0], 1).to(op_device)

                d = PreMultR(match_per_layer[layer_id], nr)
                
                Rmult = R_Multiplier(match_per_layer[layer_id])
                Pmult = P_Multiplier(match_per_layer[layer_id], nr, 1 / d)

                restriction_operators.append(Rmult)
                prolongation_operators.append(Pmult)
                restriction_for_grad_operators.append(Pmult.transpose())
                prolongation_for_grad_operators.append(Rmult.transpose())
                    
        return TransferOps(restriction_operators, prolongation_operators), TransferOps(restriction_for_grad_operators, prolongation_for_grad_operators)

