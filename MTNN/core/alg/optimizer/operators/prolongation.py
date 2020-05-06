
"""LowerTriangular Class"""
# standard
import logging
import copy
import torch
import torch.nn as nn


__all__ = ['IdentityProlongation',
           'LowerTriangleProlongation',
           'RandomSplitOperator',
           'RandomPerturbationOperator']


class IdentityProlongation:
    """Identity Interpolation Operator

    Copy model weights.
    """

    def __init__(self):
        pass

    @staticmethod
    def apply(source_model, verbose):
        return source_model


class LowerTriangleProlongation:
    """
    Transforms a model's weight matrix into a block lower triangular matrix.
    Let x be the model input with dimension N_1(row) x N_in(col)

    The Lower Triangular operator with an expansion factor of K(int) transforms the parameters of the sourcemodel,
     weight matrix W and bias vector b into a new model where each hidden layer i has a new weight matrix W' where:
     – all entries above above diagonal  W'_ij (row i, column j) are zeros.
     – all entries below the diagonal W'_ij are filled with some random noise E (float)

    Dimensions of W'_i:
    – The first hidden layer has dimensions
        – Weight matrix W'_1:  K * N_1 x N_in
        - Bias vector b'_1: K * N_1 x 1
    – The ith hidden layer has dimensions
        - Weight matrix W'_i:  K * N_i * N_(i-1)
        - Bias vector b'_i: K * N_1 X 1
    – The output layer has dimensions
        – Weight matrix W'_out: N_out X N_(i-1)
        - Bias vector b'_out: b_out X 1

     Example. In a fully connected neural network with input x = [x1, x2] and 2 hidden layers of 1 neuron each
     (a neuron being the computational unit comprising of the linear transformation and non-linear activation function).

     The parameters of the first hidden layer are the weight matrix with  W_1 = [w1_1 w1_2] and the bias vector b_1 = [b_1]
     with weights wn_ij, where n is the layer number, i is the row, j is the column.
     "                      " second hidden layer are the weight matrix, W_2 = [w2_1] and the bias vector b_2 = [b_2]

     After applying the lower triangular operator to this model, the parameters W'_i and b'_i will be:
     - Hidden layer n = 1:
        w'_1 = | w1_11   w1_12 |  b'_1 = |b_1|
               | E1_21   E1_22 |         |E_1|
     – Hidden layer n =2:
        w'_2 = | w2_11 |  b'_2 = |b_2|
               | E2_21 |         |E_2|
    Args:
        sourcemodel <MTNN.model>: a MTNN.model object
    Output:
        prolonged_model <MTNN.model>: a MTNN.model object
    """
    # TODO: Check doc string
    def __init__(self, expansion_factor: int):
        self.expansion_factor = expansion_factor

    @staticmethod
    def apply(source_model, verbose):
        """
        Takes a sourcemodel. Copies and updates weights/biases of each layer of sourcemodel into a new model with
        new block lower triangular weight matrix and augmented bias vector according to some expansion factor.
        Args:
            sourcemodel <Model>
        Returns:
            prolonged_model <Model>
        """
        # 1 Layer case:
        """
        if 
            print("SINGLE LAYER MODEL.  APPLYING IDENTITY OPERATOR...")
        return IdentityInterpolator().apply(source_model)
        # 1+ Layer case
        else:
            prolonged_model = copy.deepcopy(source_model)
        return prolonged_model
        """

class RandomPerturbationOperator:
    def __init__(self):
        pass

    def apply(self, source_model):
        pass



class RandomSplitOperator:
    def __init__(self):
        pass

    def apply(self, source_model):
        pass
