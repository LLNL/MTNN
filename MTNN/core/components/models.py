"""
Holds Models
"""
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiLinearNet",
           "BasicMnistModel",
           "BasicCifarModel"]


# API
class BaseModel(nn.Module):
    """
    Class to Overwrite
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    @abstractmethod
    def forward(self, input_):
        raise NotImplementedError

    def print(self):
        for param_tensor in self.state_dict():
            print(f"\t{param_tensor}    {self.state_dict()[param_tensor].size()}")

    def log(self, logpath):
        for param in self.parameters():
            print(param.data)
        # TODO: Write to log

############################################################################
# Implementations
##########################################################################3


class MultiLinearNet(BaseModel):
    def __init__(self, dim: list):
        """
        List of dimensions
        Args:
            dim:
        """
        super().__init__()
        modules = nn.ModuleList()

        for x in range(len(dim) - 1):
            modules.append(nn.Linear(dim[x], dim[x + 1]))

        self.layers = modules

    def forward(self, x):
        # Flatten Input
        x = x.view(x.size(0), -1)

        for idx, layer in enumerate(self.layers):
            x = self.layers[idx](x)
        x = F.relu(x)
        return x


class BasicMnistModel(BaseModel):
    """A basic image classifier."""

    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self):
        super(BasicMnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim = 1)
        return output


class BasicCifarModel(BaseModel):
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    def __init__(self):
        super(BasicCifarModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


