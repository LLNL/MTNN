"""
Copy + Noise Operator
"""

# Public API.
__all__ = ["RandomPerturbationOperator", "LowerTriangleOperator",
           "RandomSplitOperator"]

import torch


class RandomPerturbationOperator:
    def __init__(self):
       pass

    def apply(self, sourcemodel):
        pass

        # TODO: lower triangular
        # TODO: randomsplit


class LowerTriangleOperator:
    """
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
        w'_2 = | w2_11   w2_2 |  b'_2 = |b_2|
               | E2_21   E2_22|         |E_2|
    Args:
        sourcemodel <MTNN.model>: a MTNN.model object
    Output:
    """
    def __init__(self):
            pass

    def apply(self, sourcemodel):
        print("STATEDICT", sourcemodel.state_dict())
        for layer, param in enumerate(sourcemodel.state_dict()):
            print("layer", layer, "param", param)
            if "weight" or "bias" in param:
                pass
                # Transform the parameter

                # Update the parameter.
                print("StateDict\n", sourcemodel.state_dict().keys())
                print("Weights\n", sourcemodel.state_dict()["_layers.0.0.weight"].data.clone())
                test_param = torch.nn.Parameter(torch.ones(1,2))
                print(test_param)
                sourcemodel.register_parameter("prolongated_parameters", test_param)
                print("New Weight", sourcemodel.state_dict())
               # sourcemodel.state_dict[name].copy_(transformed_param)
        return

class RandomSplitOperator:
    def __init__(self):
            pass