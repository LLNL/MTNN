"""MTNN/torch_builtins.py
Stores PyTorch Functions used by Model
See: https://pytorch.org/docs/stable/optim.html#
"""
# pytorch
import torch.optim as optim
import torch.nn as nn

# Note: Keys are defined by their respective PyTorch Function Name


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
