"""
Holds Models
"""
# standard
from abc import abstractmethod

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            x = self.layers[i](x)
            x = self.activation(x)

        x = torch.flatten(x, 1)

        for i in range(self.num_conv_layers, len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[-1](x)
        x = self.output_activation(x)

        if verbose:
            printer.print_model(self, val=True)

        return x
