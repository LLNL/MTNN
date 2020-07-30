"""
Holds Models
"""
# standard
from abc import abstractmethod
from collections import namedtuple

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# local
import MTNN.utils.logger as logger
import MTNN.utils.printer as printer

log = logger.get_logger(__name__, write_to_file = True)

__all__ = ["MultiLinearNet",
           "BasicMnistModel",
           "BasicCifarModel"]


####################################################################
# Interface
###################################################################
class _BaseModel(nn.Module):
    """
    Base Model class
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    def __len__(self):
        return len(self.layers)

    @abstractmethod
    def forward(self, input_):
        """Overwrite this"""
        raise NotImplementedError

    def getGrad(self, dataloader, loss_fn, verbose=False):
        """Get gradients w.r.t to the entire dataset """
        log.debug("Get Grad")
        # Compute the gradient
        # zero the parameter gradient
        self.zero_grad()
        loss = None
        # batch loop
        for i, data in enumerate(dataloader, 0):
            # get the inputs. data is a list of [inputs, labels]
            inputs, labels = data
            if len(inputs.size()) == 4:
                inputs = inputs.view(inputs.size()[0], inputs.size()[1] * inputs.size()[2] * inputs.size()[3])
            # forward: get the loss w.r.t this batch

            outputs = self(inputs)
            print(F"{inputs = }, {outputs = }")

            if loss is None:
                loss = loss_fn(outputs, labels)
            else:
                loss += loss_fn(outputs, labels)

            #loss += loss_fn(outputs, labels)

        # TODO: Add L2 regularization printing option?
        """
            # L2 regularization
            nbatches = len(dataloader)
            if loss_fn.l2_decay != 0.0:
                l2_reg = None
                for W in self.parameters():
                    if l2_reg is None:
                        l2_reg = torch.pow(W, 2).sum()
                    else:
                        l2_reg += torch.pow(W, 2).sum()
                loss += 0.5 * nbatches * loss_fn.l2_decay * l2_reg
        """
        print(f"Model.getgrad: Total loss{loss = }")
        loss.backward()
        printer.printModel(self, msg = "MODELS.Getgrad:after update", grad = True)

        # TODO
        """
        if verbose:
            printer.printGradNorm(loss)
        """

        weight_grad, bias_grad = [], []
        for layer_id in range(len(self.layers)):
            weight_grad.append(np.copy(self.layers[layer_id].weight.grad.detach()))
            bias_grad.append(np.copy(self.layers[layer_id].bias.grad.detach()))
        Grad = namedtuple('grad', ['weight_grad', 'bias_grad'])
        grad = Grad(weight_grad, bias_grad)
        return grad

    def print(self, mode='light'):
        # TODO: Improve modality
        assert mode in ('light', 'med', 'high')
        if mode == 'light':
            for param_tensor in self.state_dict():
                log.info(f"\t{param_tensor}  {self.state_dict()[param_tensor].size()}")
        if mode == 'med':
            for layer_idx, layer in enumerate(self.layers):
                log.info(f"LAYER: {layer_idx}")
                log.info(f"\tWEIGHTS {layer.weight}\n\t BIAS{layer.bias}")
        if mode == 'high':
            for layer_idx, layer in enumerate(self.layers):
                log.info(f"LAYER: {layer_idx}")
                log.info(f" \tWEIGHTS {layer.weight}\n\tBIAS {layer.bias}")
                log.info(f" \tWEIGHT GRADIENTS {layer.weight.grad}\n\tBIAS GRADIENTS {layer.bias.grad}")


    def log(self, logpath):
        for param in self.parameters():
            print(param.data)
        # TODO: Write to log


############################################################################
# Implementations
############################################################################
class MultiLinearNet(_BaseModel):
    def __init__(self, dim: list, activation, output_activation, weight_fill=None, bias_fill=None): # Check activationtype
        """
        Builds a fully connected network given a list of dimensions
        Args:
            dim: <list> List of dimensions [dim_in, hidden ,..., dim_out]
            activation: <torch.nn.Functional> Torch activation function
            weight_fill: <float> For debugging. Value to fill weights for each layer
            bias_fill: <float> For debugging. Value to fill bias for each layer
        """
        super().__init__()
        self.activation = activation
        self.output = output_activation
        modules = nn.ModuleList()

        # Fill layers
        with torch.no_grad():
            for x in range(len(dim) - 1):
                layer = nn.Linear(dim[x], dim[x + 1])
                if weight_fill and bias_fill:
                    layer.weight.fill_(weight_fill)
                    layer.bias.fill_(bias_fill)
                modules.append(layer)

        self.layers = modules

    def forward(self, x, verbose=False):
        # Flatten Input
        x = x.view(x.size(0), -1)

        for idx, layer in enumerate(self.layers):
            if idx != (len(self.layers) - 1):
                x = self.layers[idx](x)
                x = self.activation(x)

            elif layer == self.layers[-1]:
                x = self.layers[idx](x)
                x = self.output(x, dim=1)

        if verbose:
            printer.printModel(self, val=True)

        return x

class BasicMnistModel(_BaseModel):
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


class BasicCifarModel(_BaseModel):
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


