""" MTNN/prolongation.py
 Prolongation operators definitions
"""

# Public API.
__all__ = ["IdentityOperator", "RandomPerturbationOperator", "LowerTriangleOperator",
           "RandomSplitOperator"]

# standard
import logging
import copy
import torch
import torch.nn as nn

# local
import MTNN.mtnn_defaults as mtnnconsts

# TODO: set-up logger
logging.basicConfig(filename=(mtnnconsts.EXPERIMENT_LOGS_FILENAME + ".log.txt"),
                    filemode='w',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


class IdentityOperator:
    """Identity Interpolation Operator

    Copy model weights.
    """

    def __init__(self):
        pass

    def apply(self, source_model):
        return source_model


class LowerTriangleOperator:
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
    def __init__(self):
        pass

    def apply(self, source_model, exp_factor):
        """
        Takes a sourcemodel. Copies and updates weights/biases of each layer of sourcemodel into a new model with
        new block lower triangular weight matrix and augmented bias vector according to some expansion factor.
        Args:
            sourcemodel <Model>
        Returns:
            prolonged_model <Model>
        """
        # 1 Layer case:
        if len(source_model._module_layers) == 1:
            print("SINGLE LAYER MODEL.  APPLYING IDENTITY OPERATOR...")
            return IdentityOperator().apply(source_model)

        else:
            prolonged_model = copy.deepcopy(source_model)
            prolonged_lastlayer_indx = (prolonged_model.num_layers - 1)
            logging.debug(f"APPLYING BLOCK TRIANGULAR OPERATOR WITH EXPANSION FACTOR of K ={exp_factor}")

            for index, layer_key in enumerate(source_model._module_layers):
                for layer in prolonged_model._module_layers[layer_key]:

                    # Check if linear layer
                    if hasattr(layer, 'weight'):
                        # Get weight matrix dimensions
                        weight_row = layer.weight.size()[0]
                        weight_col = layer.weight.size()[1]

                        # Block matrices dimensions
                        bias_dim = exp_factor * weight_row - weight_row
                        noise1_dim = ((exp_factor * weight_row) - weight_row, weight_col)  # First layer
                        noise2_dim = ((exp_factor * weight_row) - weight_row,
                                      (exp_factor * weight_col))  # Rest of layers
                        zero_dim = (weight_row, ((exp_factor * weight_col) - weight_col))
                        last_layer_zero_dim = (weight_row, ((exp_factor * weight_col) - weight_row))

                        # First hidden layer(weight matrix and bias vector):
                        if index == 0:

                            # Generate noise matrix E_21.
                            seed_tensor = torch.rand(noise1_dim, requires_grad=True)
                            noise_matrix = nn.init.kaiming_uniform_(seed_tensor, nonlinearity='relu')

                            # Generate bias noise  E_1.
                            bias_noise = nn.init.uniform_(torch.empty(bias_dim), a=-1.0, b=1.0)

                            # Update parameters.
                            with torch.no_grad():  # note: disable torch.grad else this update modifies the gradient
                                logging.debug(f"ORIGINAL WEIGHT {layer.weight.data}")
                                layer.weight.data = torch.cat((layer.weight.data, noise_matrix))

                                layer.bias.data = torch.cat((layer.bias.data, bias_noise))

                                logging.debug(f"PROLONGED WEIGHTS {layer.weight.data}")
                                logging.debug(f"PROLONGED BIAS {layer.bias.data}")

                        # For middle hidden layers(weight matrix and bias vector):
                        elif index > 0 and index != prolonged_lastlayer_indx:

                            # Generate bias noise  E_1.
                            bias_noise = nn.init.uniform_(torch.empty(bias_dim), a=-1.0, b=1.0)

                            # Generate concatenated noise matrix E_21 + E_22.
                            seed_tensor = torch.rand(noise2_dim, requires_grad=True)
                            noise_matrix = nn.init.kaiming_uniform_(seed_tensor, nonlinearity='relu')

                            # Generate zero matrix.
                            zero_matrix = nn.init.zeros_(torch.empty(zero_dim))

                            with torch.no_grad():  # note: disable torch.grad else this update modifies the gradient

                                layer.weight.data = torch.cat((layer.weight.data, zero_matrix), dim=1)
                                layer.weight.data = torch.cat((layer.weight.data, noise_matrix))
                                layer.bias.data = torch.cat((layer.bias.data, bias_noise))

                        # For last hidden layer(weight matrix and bias vector):
                        else:
                            # weight matrix [ W 0]
                            zero_matrix = nn.init.zeros_(torch.empty(last_layer_zero_dim))
                            with torch.no_grad():  # note: disable torch.grad else this update modifies the gradient
                                layer.weight.data = torch.cat((layer.weight.data, zero_matrix), dim=1)
                                # bias dim stays the same

            # Logging
            if source_model.debug:
                logging.debug("\n*****************************************************"
                              "SETTING MODEL CONFIGURATION"
                              "********************************************************")
                logging.debug(f"\nMODEL TYPE: {source_model._model_type}\
                               \nEXPECTED INPUT SIZE: {source_model._input_size}\
                               \nLAYERS: {source_model._module_layers}\
                               \nMODEL PARAMETERS:")
                for layer_idx in source_model._module_layers:
                    logging.debug(f"\n\tLAYER: {layer_idx} \
                                     \n\tWEIGHT: {source_model._module_layers[layer_idx][0].weight}\
                                     \n\tWEIGHT GRADIENTS: {source_model._module_layers[layer_idx][0].weight.grad}\
                                     \n\n\tBIAS: {source_model._module_layers[layer_idx][0].bias}\
                                     \n\tBIAS GRADIENT: {source_model._module_layers[layer_idx][0].bias.grad}")

            return prolonged_model


class RandomSplitOperator:
    def __init__(self):
        pass

    def apply(self, source_model):
        pass
    # TODO: Fill in


class RandomPerturbationOperator:
    def __init__(self):
        pass

    def apply(self, source_model):
        pass
    # TODO: Fill in

