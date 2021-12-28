# standard 
import collections as col
# torch 
import torch
# local
import MTNN.core.multigrid.scheme as mg
import MTNN.utils.logger as log
from MTNN.utils.datatypes import ParamVector, CoarseMapping, TransferOps

log = log.get_logger(__name__, write_to_file = True)

class PairwiseOpsBuilder:
    """PairwiseOpsBuilder is an object which is used to construct the actual 
    restriction and prolongation operators based on pairwise matching.

    This class constructs the actual R and P matrices, in comparison
    to the PairwiseOpsBuilder_MatrixFree class which constructs more
    efficient matrix-free representations of R and P. However, this
    class is useful for prototyping and is perhaps easier to read.

    """
    def __init__(self, weighted_projection = True):
        """Constructor.

        Either sums or adds fine parameters into coarse ones for
        restriction. The prolongation operator is constructed to be a
        projection with the restriction, ie $P$ is chosen so that $PR$ is
        a projection. More specifically, we have
        $$
        PR = B R_0^T (R_0^T B R_0)^{-1} R_0,
        $$
        where $B$ is a diagonal matrix of weights given by neuron vector norms, 
        and $R_0$ is the simple addition restriction operator.
        Depending on choices, we can either set $R = R_0$ or $R = (R_0^T B R_0)^{-1} R_0$,
        and then choose P accordingly. In this class we choose $R = R_0$.

        @param weighted_projection <bool> If true, compute
            restriction/prolongation as a projection weighted by
            vector norms. This choice ensures the projection exactly
            reconstructing two fine neurons if they point in exactly
            the same direction.

        """
        self.weighted_projection = weighted_projection
    
    def __call__(self, param_library, coarse_mapping, op_device):
        """
        @param param_library <ParamVector> The parameters associated with the network.
        @param coarse_mapping <CoarseMapping> The mapping from fine to coarse channels.
        @param op_device <torch.device> The device on which the ops matrices should reside.
        """
        
        # Each index of Prolongation/Restriction operators corresponds
        # to the layer to which it is to be applied
        restriction_operators = []
        prolongation_operators = []
        restriction_for_grad_operators = []
        prolongation_for_grad_operators = []

        with torch.no_grad():
            W_f_array, B_f_array = param_library.weights, param_library.biases
            for layer_ind in range(len(W_f_array)-1):
                nC = coarse_mapping.get_num_coarse(layer_ind)
                F2C_layer = coarse_mapping.get_F2C_layer(layer_ind)
                nF = len(F2C_layer)
                assert(W_f_array[layer_ind].shape[-2] == nF)

                currW = W_f_array[layer_ind].transpose(0, -2).flatten(1)
                currB = B_f_array[layer_ind]

                if self.weighted_projection:
                    WB = torch.cat((currW, currB), dim=1)
                    nr = torch.norm(WB, p=2, dim=1, keepdim=True).to(op_device)
                else:
                    nr = torch.ones(currW.shape[0], 1).to(op_device)
                
                R_layer = torch.zeros([nC, nF]).to(op_device)
                for i in range(nF):
                    R_layer[F2C_layer[i], i] = 1

                # Compute diagonal of $(R * nr) @ R_layer^T = R_layer
                # @ diag(nr) @ R_layer^T$, which is the summing of
                # subset of nr that are being coarsened together. Then
                # view result as column vector.
                # TODO: This is inefficient, computes a whole matrix
                # when only diagonal is needed.
                d = (R_layer * torch.reshape(nr, (1, -1)) @ torch.transpose(R_layer, 0, 1)).diagonal().reshape(-1, 1)

                # P = diag(nr) * R^T * diag(1./d)
                P_layer = (torch.transpose(R_layer, 0, 1) * nr) / d.reshape(1, -1)
                
                restriction_operators.append(R_layer)
                prolongation_operators.append(P_layer)

                # P_layer^T is used as the 'R' operator in computing tau corrections
                P_layer_tr = torch.transpose(P_layer, 0, 1)
                restriction_for_grad_operators.append(P_layer_tr)
                # R_layer^T is used as the 'P' operator in computing tau corrections
                R_layer_tr = torch.transpose(R_layer, 0, 1)
                prolongation_for_grad_operators.append(R_layer_tr)

        return TransferOps(restriction_operators, prolongation_operators), TransferOps(restriction_for_grad_operators, prolongation_for_grad_operators)

class R_Multiplier:
    """Matrix-free representation of an R matrix in the layer-by-layer
    transfer operators. This is much faster than constructing the R
    matrix directly, as most elements are 0.

    If $W$ is a weight matrix, $R W_f P$ restricts the outputs to a set of
    coarse neurons. During prolongation, $P W_c R$ prolongs the coarse
    neurons back to fine neurons.

    """
    
    def __init__(self, match):
        """@param match torch.Tensor containing, for each index, the index of
        the neuron to which it is matched for coarsening (or its own
        index if it is a singleton).
        """
        self.match = match

    def transpose(self):
        return R_Transpose_Multiplier(self.match)

    def __matmul__(self, Xf):
        """Use this operator as left-multiplication, as used during
        restriction. This works for tensors of any order, with
        restriction happening over the second-to-last dimension (ie
        rows in a matrix).

        Xc = R @ Xf

        """
        nf = self.match.size()[0]
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xc = Xf[..., pair1, :] + Xf[..., pair2, :]
        # for singletons, average to keep the L1 norm of the weight matrix the same
        Xc[..., pair1 == pair2, :] /= 2
        return Xc

    def __rmatmul__(self, Xc):
        """ Use this operator as right-multiplication, as used during prolongation.

        Xf = Xc @ R
        """
        nf = self.match.size()[0]
        Xf = torch.empty((*Xc.size()[:-1], nf), device=Xc.device)
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xf[..., pair2] = Xc
        Xf[..., pair1] = Xc
        return Xf


class P_Multiplier:
    """Matrix-free representation of a P matrix in the layer-by-layer
    transfer operators. This is much faster than constructing the P
    matrix directly, as most elements are 0.

    If $W$ is a weight matrix, $R W_f P$ restricts the outputs to a set of
    coarse neurons. During prolongation, $P W_c R$ prolongs the coarse
    neurons back to fine neurons.

    """

    def __init__(self, match, d1, d2):
        """
        @param match torch.Tensor containing, for each index, the index of
        the neuron to which it is matched for coarsening (or its own
        index if it is a singleton).

        @param d1 The vector of diagonal elements to left-scale R^T as with $D1 @ R^T @ D2$

        @param d2 The vector of diagonal elements to right-scale R^T as with $D1 @ R^T @ D2$
        """
        self.match = match
        self.d1 = d1
        self.d2 = d2

    def transpose(self):
        return P_Transpose_Multiplier(self.match, self.d1, self.d2)

    def __matmul__(self, Xc):
        """Use this operator as left-multiplication, as used during
        prolongation.

        Xf = P @ Xc = D1 @ R^{T} @ D2 @ Xc

        """
        nf = self.match.size()[0]
        Xf = torch.empty((*Xc.shape[:-2], nf, Xc.shape[-1]), device=Xc.device)
        pair2 = self.match[self.match >= torch.arange(0, nf, device=self.match.device)]
        pair1 = self.match[pair2]
        Xc = Xc * self.d2.reshape(-1, 1) # TODO: Is this correct for higher order tensors?
        Xf[..., pair2, :] = Xc
        Xf[..., pair1, :] = Xc
        Xf = Xf * self.d1.reshape(-1, 1) # TODO: Is this correct for higher order tensors?
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
    """Matrix-free representation of the tranpose of an R matrix in the
    layer-by-layer transfer operators. Significantly more efficient
    than constructing $R^T$ directly, as most elements are 0.

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
    """Matrix-free representation of the transpose of a P matrix in the
    layer-by-layer transfer operators. Significantly more efficient
    than constructing $P^T$ directly, as most elements are 0.

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
    """PairwiseOpsBuilder_MatrixFree is an object which is used to
    construct the actual restriction and prolongation operators based
    on pairwise matching. This is similar to PairwiseOpsBuilder except
    that it constructs matrix-free versions of the operators which are
    much more efficient than the actual-matrix version, which are
    mostly 0s.

    """
    def __init__(self, weighted_projection = True):
        """Constructor.

        Either sums or adds fine parameters into coarse ones for
        restriction. The prolongation operator is constructed to be a
        projection with the restriction, ie $P$ is chosen so that $PR$ is
        a projection. More specifically, we have
        $$
        PR = B R_0^T (R_0^T B R_0)^{-1} R_0,
        $$
        where $B$ is a diagonal matrix of weights given by neuron
        vector norms, and $R_0$ is the simple addition restriction
        operator.  Depending on choices, we can either set $R = R_0$
        or $R = (R_0^T B R_0)^{-1} R_0$, and then choose P
        accordingly. In this class, we choose $R = R_0$

        @param weighted_projection <bool> If true, compute restriction/prolongation 
            as a projection weighted by vector norms. This choice ensures the 
            projection exactly reconstructing two fine neurons if they point in 
            exactly the same direction.

        """
        self.weighted_projection = weighted_projection

    # Xc = R * Xf
    @staticmethod
    def PreMultR(match, Xf):
        nf = match.size()[0]
        pair2 = match[match >= torch.arange(0, nf, device=match.device)]
        pair1 = match[pair2]
        Xc = Xf[pair1, :] + Xf[pair2, :]
        # singleton
        Xc[pair1 == pair2, :] /= 2
        return Xc
    
    def __call__(self, param_library, coarse_mapping, op_device):
        """
        @param param_library <ParamVector> The parameters associated with the network.
        @param coarse_mapping <CoarseMapping> The mapping from fine to coarse channels.
        @param op_device <torch.device> The device on which the ops matrices should reside.
        """
        
        # Each index of Prolongation/Restriction operators corresponds to layer to which it is to be applied
        restriction_operators = []
        prolongation_operators = []
        restriction_for_grad_operators = []
        prolongation_for_grad_operators = []

        D2 = []

        with torch.no_grad():
            W_f_array, B_f_array = param_library.weights, param_library.biases
            for layer_ind in range(len(W_f_array)-1):
                match = coarse_mapping.get_match(layer_ind)
                nF = len(match)
                nC = coarse_mapping.get_num_coarse(layer_ind)
                assert(W_f_array[layer_ind].shape[-2] == nF)
                
                currW = W_f_array[layer_ind].transpose(0, -2).flatten(1)
                currB = B_f_array[layer_ind]

                if self.weighted_projection:
                    WB = torch.cat((currW, currB), dim=1)
                    nr = torch.norm(WB, p=2, dim=1, keepdim=True).to(op_device)
                else:
                    nr = torch.ones(currW.shape[0], 1).to(op_device)

                d = self.__class__.PreMultR(match, nr)
                
                Rmult = R_Multiplier(match)
                Pmult = P_Multiplier(match, nr, 1 / d)

                restriction_operators.append(Rmult)
                prolongation_operators.append(Pmult)
                restriction_for_grad_operators.append(Pmult.transpose())
                prolongation_for_grad_operators.append(Rmult.transpose())
                    
        return TransferOps(restriction_operators, prolongation_operators), TransferOps(restriction_for_grad_operators, prolongation_for_grad_operators)

