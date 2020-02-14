"""MTNN/torch_constants.py
Stores PyTorch Functions used by Model
See: https://pytorch.org/docs/stable/optim.html#
Note: Keys are defined by their respective PyTorch Function Name
# TODO: Make keys case-insensitive
"""
# pytorch
import torch.nn as nn


LOSS = {
    "crossentropy": nn.CrossEntropyLoss(),
    "mseloss": nn.MSELoss(),
}

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax()
}
