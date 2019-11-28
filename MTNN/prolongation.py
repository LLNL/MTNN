"""
# Prolongation operators
"""

# Public API.
__all__ = ["RandomPerturbationOperator", "LowerTriangleOperator",
           "RandomSplitOperator"]
import copy
import torch
import torch.nn as nn


class RandomPerturbationOperator:
    def __init__(self):
       pass

    def apply(self, sourcemodel):
        pass

        # TODO: lower triangular
        # TODO: randomsplit


class LowerTriangleOperator:
    """
    Transforms a model's weight matrix into a block lower triangular matrix.
    Let x be the model input with dimension N_1(row) x N_in(col)

    The Lower Triangular operator with an expansion factor of K(int) transforms the parameters of the sourcemodel,
     weight matrix W and bias vector b into a new model where each hidden layer i has a new weight matrix W'_i where:
     – all entries above above diagonal  W'_ij (row i, column j) are zeros.
     – all entries below the diagonal W'_ij are filled with some random noise E (float)

    Dimensions of W'_i
    – The first hidden layer has dimensions
        – Weight matrix W'_1:  K * N_1 x N_in
        - Bias vector b'_1: K * N_1 x 1
    – The ith hidden layer has dimensions
        - Weight matrix W'_i:  K * N_i * N_(i-1)
        - Bias vector b'_i: K * N_1 X 1
    – The output layer has dimensions
        – Weight matrix W'_out: N_out X N_(i-1)
        - Bias vector b'_out: b_out X 1

     Ex. In a fully connected neural network with input x = [x1, x2] and 2 hidden layers of 1 neuron each (aka perceptron)
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
    def __init__(self):
        pass

    def apply(self, sourcemodel, expansionfactor):
        """
        Takes a sourcemodel. Copies and updates weights/biases of each layer of sourcemodel into a new model.
        Args:
            sourcemodel:

        Returns:

        """
        print("\nCopying model")
        prolonged_model = copy.deepcopy(sourcemodel)

        # Keep track of statedict?
        print("\nSOURCE\n", sourcemodel.view_parameters())
        print("\nCURRENT STATEDICT\n", sourcemodel.state_dict())
        #print(sourcemodel.parameters())


        #print(prolonged_model.parameters())
        print("\nNew\n",prolonged_model.view_parameters())
        print("\nNEW STATEDICT\n", prolonged_model.state_dict())
        #print(prolonged_model._layers)

        # Permute depending on the layer
        print("\nAPPLYING LOWER TRIANGULAR OPERATOR")
        for index, layer_key in enumerate(sourcemodel._layers):
            for layer in prolonged_model._layers[layer_key]:
                # Check linear layer
                if hasattr(layer, 'weight'):
                    print(layer)
                    print(index, sourcemodel._num_layers)
                    #print("weight", layer.weight)
                    #print("weight0", layer.weight[0])
                    weight_dim = layer.weight.size()
                    print(weight_dim)

                    # First hidden layer
                    if index == 0:
                        # Generate noise matrix E_21
                        #weight_size = layer.weight[0].size() #scalar/0d tensor
                        random_tensor = torch.rand(weight_dim)
                        print("Random tensor", random_tensor)
                        #error = torch.rand(1)#Overwrite Kaiming_uniform? Doesn't work with tensors <2 dim
                        noise_matrix = nn.init.kaiming_uniform_(random_tensor, nonlinearity='relu')
                        #print("Error", error.item()) #error.item() converts one element tensors to python scalars
                        print("Weight Noise", noise_matrix)
                        bias_size = layer.bias.size()
                        bias_noise = nn.init.uniform_(torch.empty(bias_size), a=-1.0, b=1.0)
                        print("Bias Noise", bias_noise)

                        # Temporarily set disable gradient calculation of weights/biases; update weights only
                        # Using .data is not recommended; disables autograd from throwing errors
                        with torch.no_grad():
                            print("Weight matrix:", layer.weight.data)
                            print("Bias:", layer.bias.data)

                            #layer.weight.data.fill_(error) # fill only supports 0d dimension, disables autograd warning
                            layer.weight.data = torch.cat((layer.weight.data, noise_matrix))
                            layer.bias.data = torch.cat((layer.bias.data, bias_noise))

                            print("New weight matrix:", layer.weight.data)
                            print("New bias", layer.bias.data)
                        #layer.weight.data(error) # Runtime error: a leaf Variable that requires grad has been used in an inplace operation
                        #layer.weight.data[0].fill_(error.item()) # fill_ only takes 0d tensor

                    # Rest of hidden layers except for output layer
                    elif index > 0:
                        # Generate noise matrix E_21
                        random_tensor = torch.rand(weight_dim)
                        #print("Random tensor", random_tensor)
                        noise1 = nn.init.kaiming_uniform_(random_tensor, nonlinearity = 'relu')
                        print("\nNOISE1", noise1)

                        # Generate noise matrix E_22
                        random_tensor2 = torch.rand(weight_dim)
                        #print("Random tensor", random_tensor2)
                        noise2 = nn.init.kaiming_uniform_(random_tensor, nonlinearity = 'relu')
                        print("\nNOISE2", noise2)

                        noise_matrix = torch.cat((noise1, noise2), dim =1)
                        print("\nWEIGHT NOISE", noise_matrix)

                        # Generate zero matrix
                        zero_matrix = nn.init.zeros_(torch.empty(weight_dim))
                        print("\nZero matrix:\n", zero_matrix)

                        # Generate bias noise
                        bias_size = layer.bias.size()
                        bias_noise = nn.init.uniform_(torch.empty(bias_size), a = -1.0, b = 1.0)
                        print("Bias Noise", bias_noise)

                        with torch.no_grad():
                            print("Weight matrix:", layer.weight.data)
                            print("Bias:", layer.bias.data)

                            layer.weight.data = torch.cat((layer.weight.data, zero_matrix), dim=1)
                            layer.weight.data = torch.cat((layer.weight.data, noise_matrix))
                            layer.bias.data = torch.cat((layer.bias.data, bias_noise))

                            print("New weight matrix:", layer.weight.data)
                            print("New bias", layer.bias.data)


                    """
                    print(layer.weight.data)
                    layer.bias.data.fill_(0.5)
                    print(layer.bias.data)
                    """
        #print("Dir:", sourcemodel.__dir__)
        #print(dir(sourcemodel))
        #print("\nSOURCE\n", sourcemodel.view_parameters())
        #print("\nPROLONGED\n", prolonged_model.view_parameters())


        """
        for layer, param in enumerate(prolonged_model.state_dict()):
            print("layer", layer, "param", param)
            if "weight" or "bias" in param:
                pass
                # Transform the parameter
                param.data.fill_(1.0)
                # Update the parameter.
                #print("StateDict\n", sourcemodel.state_dict().keys())
                #print("Weights\n", sourcemodel.state_dict()["_layers.0.0.weight"].data.clone())
                test_param = torch.nn.Parameter(torch.ones(1,2))
                #print(test_param)
                sourcemodel.register_parameter("prolongated_parameters", test_param)
                #print("New Weight", sourcemodel.state_dict())
               # sourcemodel.state_dict[name].copy_(transformed_param)
        """

        return prolonged_model

class RandomSplitOperator:
    def __init__(self):
            pass