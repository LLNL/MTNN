"""
Torch Activation Function Dispatch Table
See documentation here:
# TODO: Fill in as needed
"""

# PyTorch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax()
}
