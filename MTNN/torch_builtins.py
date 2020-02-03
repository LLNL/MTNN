"""MTNN/torch_builtins.py
Stores PyTorch Functions and Variables
See: https://pytorch.org/docs/stable/optim.html#
"""
# pytorch
import torch.optim as optim
import torch.nn as nn

# Note: Keys are defined by their respective PyTorch Function Name

OPTIMIZATIONS = {
    "Adadelta": optim.Adadelt(),
    "Adagrad": optim.Adagrad(),
    "Adam": optim.Adam(),
    "AdamW": optim.AdamW(),
    "SparseAdam": optim.SparseAdam(),
    "Adamax": optim.Adamax(),
    "ASGD": optim.ASGD(),
    "LBFGS": optim.LBFGS(),
    "RMSprop": optim.RMSprop(),
    "Rprop": optim.Rprop(),
    "SGD": optim.SGD()
}

LOSS = {
    "crossentropy": nn.CrossEntropyLoss(),
    "mseloss": nn.MSELoss()
}

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax()
}
