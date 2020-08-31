"""
Prolongation operators
"""
# standard
import copy
import numpy as np
from abc import ABC, abstractmethod


# third-party
import torch
import torch.nn as nn

# local
from MTNN.utils import logger, printer

log = logger.get_logger(__name__, write_to_file = True)

__all__ = ['IdentityProlongation',
           'LowerTriangleProlongation',
           'PairwiseAggProlongation'
           ]


####################################################################
# Interface
###################################################################
class _BaseProlongation(ABC):
    """Overwrite this"""

    @abstractmethod
    def apply(self, **kwargs):
        raise NotImplementedError


###################################################################
# Implementation
####################################################################
class IdentityProlongation(_BaseProlongation):
    """Identity Interpolation Operator

    Copy model weights.
    """

    def __init__(self):
        pass

    @staticmethod
    def apply(source_model, verbose):
        return source_model


class LowerTriangleProlongation(_BaseProlongation):
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
        – Weight matrix W'_out: N_out * N_(i-1)
        - Bias vector b'_out: b_out * 1

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

    def apply(self, fine_level, coarse_level, dataloader, verbose=False):
        """
        Takes a sourcemodel. Copies and updates weights/biases of each layer of sourcemodel into a new model with
        new block lower triangular weight matrix and augmented bias vector according to some expansion factor.
        Args:
            sourcemodel <MTNN.BaseModel>
        Returns:
            prolonged_model <MTNN.BaseModel>
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

        # TODO: Add Check for fully connected layers
        # Single Hidden Layer
        if len(fine_level.net.layers) <= 2:
            return fine_level

        # Multi-hidden layer
        else:
            coarse_level.net = copy.deepcopy(fine_level.net)
            last_layer_idx = len(coarse_level.net.layers) - 1

            for index, layer in enumerate(coarse_level.net.layers):

                # Get weight matrices dimensions
                weight_row = layer.weight.size()[0]
                weight_col = layer.weight.size()[1]

                # Block matrices dimensions
                bias_dim = self.expansion_factor * weight_row - weight_row
                noise1_dim = ((self.expansion_factor * weight_row) - weight_row, weight_col)  # First layer
                noise2_dim = ((self.expansion_factor * weight_row) - weight_row,
                              (self.expansion_factor * weight_col))  # Rest of layers
                zero_dim = (weight_row, ((self.expansion_factor * weight_col) - weight_col))
                last_layer_zero_dim = (weight_row, ((self.expansion_factor * weight_col) - weight_col))

                # First hidden layer(weight matrix and bias vector):
                if index == 0:

                    # Generate noise matrix E_21.
                    seed_tensor = torch.rand(noise1_dim, requires_grad = True)
                    noise_matrix = nn.init.kaiming_uniform_(seed_tensor, nonlinearity = 'relu')

                    # Generate bias noise  E_1.
                    bias_noise = nn.init.uniform_(torch.empty(bias_dim), a = -1.0, b = 1.0)

                    # Update parameters.
                    with torch.no_grad():  # note: disable torch.grad else this update modifies the gradient

                        layer.weight.data = torch.cat((layer.weight.data, noise_matrix))
                        layer.bias.data = torch.cat((layer.bias.data, bias_noise))


                # For middle hidden layers(weight matrix and bias vector):
                elif index > 0 and index != last_layer_idx:

                    # Generate bias noise  E_1.
                    bias_noise = nn.init.uniform_(torch.empty(bias_dim), a = -1.0, b = 1.0)

                    # Generate concatenated noise matrix E_21 + E_22.
                    seed_tensor = torch.rand(noise2_dim, requires_grad = True)
                    noise_matrix = nn.init.kaiming_uniform_(seed_tensor, nonlinearity = 'relu')

                    # Generate zero matrix.
                    zero_matrix = nn.init.zeros_(torch.empty(zero_dim))

                    with torch.no_grad():  # note: disable torch.grad else this update modifies the gradient

                        layer.weight.data = torch.cat((layer.weight.data, zero_matrix), dim = 1)
                        layer.weight.data = torch.cat((layer.weight.data, noise_matrix))
                        layer.bias.data = torch.cat((layer.bias.data, bias_noise))


                # For last hidden layer(weight matrix and bias vector):
                else:
                    # weight matrix [ W 0]
                    zero_matrix = nn.init.zeros_(torch.empty(last_layer_zero_dim))
                    with torch.no_grad():  # note: disable torch.grad else this update modifies the gradient

                        layer.weight.data = torch.cat((layer.weight.data, zero_matrix), dim = 1)
                        # bias dim stays the same

        if verbose:
            printer.print_model(fine_level.net, msg="ORIGINAL",  val=True, dim=True)
            printer.print_model(coarse_level.net, msg="PROLONGED", val=True, dim=True)



class PairwiseAggProlongation(_BaseProlongation):
    def __init__(self):
        self.prolongation_operators = []

    def apply(self, fine_level, coarse_level, dataloader, verbose):
        """
        Args:
            fine_level:
            coarse_level:
            dataloader:
            verbose:

        Returns:

        """
        # Check PairwiseAggSetup
        assert fine_level.interpolation_data is not None
        num_coarse_layers = len(coarse_level.net.layers)
        R_array, P_array = [*fine_level.interpolation_data]

        for layer_id in range(num_coarse_layers):
            W_c = coarse_level.net.layers[layer_id].weight.detach().clone()
            B_c = coarse_level.net.layers[layer_id].bias.detach().clone().reshape(-1, 1)
            e_W = W_c - coarse_level.Winit[layer_id]  # W_c = self.R_array[layer_id] @ W_f @ self.P_array[layer_id-1]
            e_B = B_c - coarse_level.Binit[layer_id]
            if layer_id < num_coarse_layers - 1:
                if layer_id == 0:
                    e_W = P_array[layer_id] @ e_W
                else:
                    e_W = P_array[layer_id] @ e_W @ R_array[layer_id - 1]
                e_B = P_array[layer_id] @ e_B
            elif layer_id > 0:
                e_W = e_W @ R_array[layer_id - 1]

            W_f = fine_level.net.layers[layer_id].weight.detach().clone()
            B_f = fine_level.net.layers[layer_id].bias.detach().clone().reshape(-1, 1)
            W_f += e_W
            B_f += e_B

            # Update fine-level net
            with torch.no_grad():
                fine_level.net.layers[layer_id].weight.copy_(W_f)
                fine_level.net.layers[layer_id].bias.copy_(B_f.reshape(-1))

class RandomPerturbationOperator(_BaseProlongation):
    def __init__(self):
        pass

    def apply(self, source_model):
        pass


class RandomSplitOperator(_BaseProlongation):
    def __init__(self):
        pass

    def apply(self, source_model):
        pass
