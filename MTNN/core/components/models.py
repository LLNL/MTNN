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
from MTNN.utils import logger, printer, deviceloader

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
        self.device = deviceloader.get_device(verbose=False)

    def __len__(self):
        return len(self.layers)

    @abstractmethod
    def forward(self, input_):
        """Overwrite this"""
        raise NotImplementedError

    def getGrad(self, dataloader, loss_fn, verbose=False):
        """Get gradients w.r.t to the entire dataset """
        # Compute the gradient
        self.zero_grad()   # zero the parameter gradient
        total_loss = None
        for i, mini_batch_data in enumerate(dataloader, 0):
            # get the inputs. data is a list of [inputs, labels]
            inputs, labels = deviceloader.load_data(mini_batch_data, self.device)

            if len(inputs.size()) == 4:
                inputs = inputs.view(inputs.size()[0], inputs.size()[1] * inputs.size()[2] * inputs.size()[3])

            # forward: get the loss w.r.t this batch
            outputs = self(inputs)

            if total_loss is None:
                total_loss = loss_fn(outputs, labels)
            else:
                total_loss += loss_fn(outputs, labels)

        total_loss.backward()

        # TODO: Print L2 regularization?
        """
        if verbose:
            printer.printGradNorm(loss)
        """

        weight_grad, bias_grad = [], []
        for layer_id in range(len(self.layers)):
            weight_grad.append(self.layers[layer_id].weight.grad.detach().clone())
            bias_grad.append(self.layers[layer_id].bias.grad.detach().clone())
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

    def set_device(self, device): 
        self.layers.to(device)

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
            output_activation: <torch.nn.Functional or lambda> Final function.
            weight_fill: <float> For debugging. Value to fill weights for each layer
            bias_fill: <float> For debugging. Value to fill bias for each layer
        """
        super().__init__()
        self.activation = activation
        self.output = output_activation
        self.dim = dim
       
        # Fill layers
        modules = nn.ModuleList()
        with torch.no_grad():
            for x in range(len(dim) - 1):
                layer = nn.Linear(dim[x], dim[x + 1])
                if weight_fill and bias_fill:
                    layer.weight.fill_(weight_fill)
                    layer.bias.fill_(bias_fill)
                modules.append(layer)

        self.layers = modules
        self.layers.to(self.device) 

    def forward(self, x, verbose=False):
        # Flatten Input
        x = x.view(x.size(0), -1)

        for idx, layer in enumerate(self.layers):
            if idx != (len(self.layers) - 1):
                x = self.layers[idx](x)
                x = self.activation(x)

            elif layer == self.layers[-1]:
                x = self.layers[idx](x)
                x = self.output(x)

        if verbose:
            printer.print_model(self, val=True)

        return x

    def all_hidden_forward(self, x):
        # Flatten Input
        x = x.view(x.size(0), -1)

        outputs = [x.detach().clone()]
        with torch.no_grad():
            for idx, layer in enumerate(self.layers):
                if idx != (len(self.layers) - 1):
                    x = self.layers[idx](x)
                    x = self.activation(x)
                elif layer == self.layers[-1]:
                    x = self.layers[idx](x)
                    x = self.output(x)
                outputs.append(x.detach().clone())

        return outputs

class ConvolutionalNet(_BaseModel):
    def __init__(self, conv_channels: list, fc_dims: list, activation, output_activation):
        """
        Builds a network of several convolutional layers followed by several fully-connected layers.

        Args:
            conv_channels: <List(out_channels, kernel_width, stride)> List of convolutional channel 
                           information. The first layer is the input layer, for which kernel_width 
                           and stride are ignored.
            fc_dims: <List> List of fully-connected dimensions. The first elemenet is the number 
                            of DOFs coming out of the final convolutional layer.
            activation: <torch.nn.Functional or lambda> Activation function.
            output_activation: <torch.nn.Functional or lambda> Final function.
        """
        super().__init__()

        self.activation = activation
        self.output_activation = output_activation
        self.conv_channels = conv_channels
        self.num_conv_layers = len(conv_channels)-1
        self.fc_dims = fc_dims

        # Fill layers
        modules = nn.ModuleList()
        with torch.no_grad():
            for i in range(self.num_conv_layers):
                in_ch, _, _ = self.conv_channels[i]
                out_ch, kernel_width, stride = self.conv_channels[i+1]
                layer = nn.Conv2d(in_ch, out_ch, kernel_width, stride)
                modules.append(layer)
            for i in range(1, len(self.fc_dims)):
                layer = nn.Linear(self.fc_dims[i-1], self.fc_dims[i])
                modules.append(layer)

        self.layers = modules
        self.layers.to(self.device)

    def forward(self, x, verbose=False):
        for i in range(self.num_conv_layers):
#            print(i, x.shape)
            x = self.layers[i](x)
            x = self.activation(x)

#        print("After conv", x.shape)
        x = torch.flatten(x, 1)
#        print("Flattened", x.shape)
        for i in range(self.num_conv_layers, len(self.layers)-1):
#            print(i, x.shape)
            x = self.layers[i](x)
            x = self.activation(x)

#        print(i+1, x.shape)
        x = self.layers[-1](x)
        x = self.output_activation(x)

        if verbose:
            printer.print_model(self, val=True)

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


