"""
Copy + Noise Operator
"""
# Public API
__all__ = ["Random_perturbation_operator", "Lower_triangular_operator",
           "Random_split_operator"]

import torch


class Random_perturbation_operator:
    def __init__(self):
       pass

    def apply(self, sourcemodel):
        for name, param in enumerate(sourcemodel.state_dict()):
            if "weight" in param:
                pass
                # Transform the parameter
                print(sourcemodel.parameters())
                # Update the parameter.
               # sourcemodel.state_dict[name].copy_(transformed_param)

        # TODO: lower triangular
        # TODO: randomsplit

class Lower_triangular_operator:
    """
    Let x be the model input with dimension N_1(row) x N_in(col)

    The Lower Triangular operator with an expansion factor of K(int) transforms the parameters of the sourcemodel,
     weight matrix W and bias vector b into a new model where each hidden layer i has a new weight matrix W'_i where:
     – all entries above above diagonal  W'_ij (row i, column j) are zeros.
     – all entries below the diagonal W'_ij are filled with some random noise E (float)

    Dimensions of W'_i
    – The first hidden layer has dimensions
        – Weight matrix W'_1:  K * N_1 x N_in
        - Bias vector b'_1: K * N_1 x N_1
    – The ith hidden layer has dimensions
        - Weight matrix W'_i:  K * N_i * N_(i-1)
        - Bias vector b'_i: K * N_1 X N_1
    – The output layer has dimensions
        – Weight matrix W'_out: N_out X N_(i-1)
        - Bias vector b'_out: b_out

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
        w'_2 = | w2_11   w2_2 |  b'_2 = |b_2|
               | E2_21   E2_22|         |E_2|
    """
    def __init__(self):
            pass

    def apply(self, sourcemodel):
        print("STATEDICT", sourcemodel.state_dict())
        for name, param in enumerate(sourcemodel.state_dict()):
            print("key, value", name, param)
            if "weight" or "bias" in param:
                pass
                # Transform the parameter
                print("Parameters", param)
                # Update the parameter.
                print("StateDict", sourcemodel.state_dict().keys())
                print("Weights", sourcemodel.state_dict()["layers.0.0.weight"].data.clone())
                test_param = torch.nn.Parameter(sourcemodel.state_dict()["layers.0.0.weight"].data.clone())
                print(test_param)
                sourcemodel.register_parameter("prolongated_parameters", test_param)
               # sourcemodel.state_dict[name].copy_(transformed_param)
        return

class Random_split_operator:
    def __init__(self):
            pass