"""Holds namedtuples definitions"""
from collections import namedtuple
from functools import singledispatch
import torch

############################################
# Data types
#############################################

rhs = namedtuple('rhs', ['W', 'b'])
level_data = namedtuple('level_data', ['R', 'P', 'W', 'b', 'rhsW', 'rhsB'])
operators = namedtuple("operators", "R_op P_op R_for_grad_op P_for_grad_op, l2reg_left_vecs, l2reg_right_vecs")


class CoarseMapping:
    """CoarseMapping - specifies a mapping from fine channels to coarse channels.
    
    num_coarse_channels <List> - A list of length equal to the number
    of network layers. Each element contains the number of coarse
    neurons (num channels in CNNs) in that layer.

    match_per_layer <List(List)> - A list of lenght equal to the
    number of layers. Each element is itself a list, containing the
    neuron to which it is matched for coarsening.

    """
    def __init__(self, num_coarse_channels, match_per_layer):
        self.num_coarse_channels = num_coarse_channels
        self.match_per_layer = match_per_layer

    def get_num_coarse(self, layer_ind):
        return self.num_coarse_channels[layer_ind]

    def get_match(self, layer_ind):
        return self.match_per_layer[layer_ind]

    def get_F2C_layer(self, layer_ind):
        match = self.match_per_layer[layer_ind]
        F2C_layer = torch.zeros(match.shape[0], dtype=int)
        curr_coarse_ind = 0
        for i, m_ind in enumerate(match):
            if m_ind < i:
                F2C_layer[i] = F2C_layer[m_ind]
            else:
                F2C_layer[i] = curr_coarse_ind
                curr_coarse_ind += 1
        return F2C_layer


class TransferOps:
    """TransferOps - The matrix operators used in restriction and prolongation.
    
    R_ops <List> - A list, of length equal to the number of layers minus
    1, containing the R operators used in restriction.
    
    P_ops <List> - A list, of length equal to the number of layers minus
    1, containing the P operators used in restriction.
    """
    def __init__(self, R_ops, P_ops):
        self.R_ops = R_ops
        self.P_ops = P_ops

    def swap_transfer_ops(self):
        """Swap operator lists .  
        This is used to convert a restriction
        operator into a prolongation and vice versa.

        Inputs:
        None
        
        Output:
        (TransferOps) The operator-swapped operator.
        """
        return TransferOps(self.P_ops, self.R_ops)

class ParamVector:
    """ParamVector - a data store of parameters associated with a neural network

    weights_list and bias_list are each lists of length equal to the
    number of layers in a network.

    Each element of weights_list is a tensor of order >=2 such that
    dimension -2 is the number of output channels and dimension -1 is the
    number of input channels. Note that for fully-connected layers, this
    tensor will be of order 2 exactly, in which case dimension -2 is the
    rows and dimension -1 is the columns.

    Each element of bias_list is a tensor of order 1 of length equal to
    the number of output channels.
    """

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.num_layers = len(self.weights)
        assert(len(weights) == len(biases))

    def __neg__(self):
        new_weights = []
        new_biases = []
        for layer_id in range(self.num_layers):
            new_weights.append(-self.weights[layer_id])
            new_biases.append(-self.biases[layer_id])
        return ParamVector(new_weights, new_biases)
        

    def __add__(self, other):
        if other == 0:
            return self
        new_weights = []
        new_biases = []
        for layer_id in range(self.num_layers):
            new_weights.append(self.weights[layer_id] + other.weights[layer_id])
            new_biases.append(self.biases[layer_id] + other.biases[layer_id])
        return ParamVector(new_weights, new_biases)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        if other == 0:
            return self
        new_weights = []
        new_biases = []
        for layer_id in range(self.num_layers):
            new_weights.append(self.weights[layer_id] - other.weights[layer_id])
            new_biases.append(self.biases[layer_id] - other.biases[layer_id])
        return ParamVector(new_weights, new_biases)

    def __rsub__(self, other):
        if other == 0:
            return -self
        new_weights = []
        new_biases = []
        for layer_id in range(self.num_layers):
            new_weights.append(other.weights[layer_id] - self.weights[layer_id])
            new_biases.append(other.weights[layer_id] - self.biases[layer_id])
        return ParamVector(new_weights, new_biases)
        
    def __rmatmul__(self, R):
        """ Compute the matrix-vector multiplication R @ v.

        If R is a restriction operator R, then Rv produces restricted parameters.
        If R is a prolongation operator P, then Pv produces prolonged parameters
        This was previously called the 'transfer' function.

        Inputs:
        R (TransferOps) - The TransferOps objects representing the big R matrix.
        """
        # Matrix multiplication broadcasting:
        # T of shape (a, b, c, N, M)
        # Matrix A of shape (k, N)
        # Matrix B of shape (M, p)
        # A @ T @ B computes a tensor of shape (a, b, c, k, p) such that
        # (i, j, l, :, :) = A @ T(i,j,l,:,:) @ B
        def mul(A, B):
            try:
                # works for anything except where first param is
                # Tensor and second param isn't...
                return A @ B
            except TypeError:
                # ...in which case a TypeError gets thrown and we call this
                return B.__rmatmul__(A)
        Wdest_array = []
        Bdest_array = []
        for layer_id in range(self.num_layers):
            Wsrc = self.weights[layer_id]
            Bsrc = self.biases[layer_id]
            if layer_id < self.num_layers - 1:
                if layer_id == 0:
                    Wdest = mul(R.R_ops[layer_id], Wsrc)
                else:
                    Wdest = mul(mul(R.R_ops[layer_id], Wsrc), R.P_ops[layer_id - 1])
                Bdest = mul(R.R_ops[layer_id], Bsrc)
            elif layer_id > 0:            
                Wdest = mul(Wsrc, R.P_ops[layer_id-1])
                Bdest = Bsrc.clone()

            Wdest_array.append(Wdest)
            Bdest_array.append(Bdest)
        return ParamVector(Wdest_array, Bdest_array)

