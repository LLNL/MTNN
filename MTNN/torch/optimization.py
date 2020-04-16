"""
Torch Optimization Function Dispatch Table
See documentation: <https://pytorch.org/docs/stable/optim.html>
# TODO: Fill in as needed
"""
# PyTorch
import torch.optim as optim

def Adadelta():
    pass


def Adagrad():
    pass


def Adam():
    pass


def AdamW():
    pass


def SparseAdam():
    pass


def Adamax():
    pass


def ASGD():
    pass


def LBFGS():
    pass


def RMSprop():
    pass


def Rprop():
    pass


def SGD(model_parameters, learning_rate, momentum):
    """
    Create and return Torch stochastic gradient descent optimizer
    Args:
        model_parameters:
        learning_rate:
        momentum:

    Returns:

    """
    opt = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum)
    return opt