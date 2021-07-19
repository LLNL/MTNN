"""
Restriction Operators
"""
import torch
import torch.nn as nn
import numpy as np
import collections as col
import copy
from abc import ABC, abstractmethod

# local
import MTNN.core.multigrid.scheme as mg
import MTNN.core.components.models as models
from MTNN.utils.datatypes import operators
import MTNN.utils.logger as log
import MTNN.utils.printer as printer
import MTNN.utils.deviceloader as deviceloader
from MTNN.core.multigrid.operators.SecondOrderRestriction import ParamLibrary

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['SecondOrderConverter',
           'MultilinearConverter',
           'ConvolutionalConverter']

############################################
# Extractors
#############################################

class ParameterExtractor:
    """ParameterExtractor.

    This class extracts parameter tensors from a neural network to a
    ParamLibrary for restriction/prolongation processing.
    """
    def __init__(self, converter):
        self.converter = converter
    
    def extract_from_network(self, level):
        """Pull parameters out of a network.

        Input: <Level>.

        Output: <tuple(ParamLibrary, ParamLibrary)>. The first
        ParamLibrary represents network parameters. The second
        represents momentum values.

        """
        net = level.net
        optimizer = level.presmoother.optimizer

        # Pull parameters from the network
        W_array = [net.layers[layer_id].weight.detach() for layer_id in range(len(net.layers))]
        B_array = [net.layers[layer_id].bias.detach().reshape(-1, 1) for layer_id in range(len(net.layers))]

        # Pull momentum values from the optimizer.
        get_p = lambda ind : optimizer.state[optimizer.param_groups[0]['params'][ind]]['momentum_buffer']
        mW_array, mB_array = [list(x) for x in zip(*[(get_p(2*i), get_p(2*i+1).reshape(-1, 1)) for i in
                                                     range(int(len(optimizer.param_groups[0]['params']) / 2))])]

        param_library, momentum_library = (ParamLibrary(W_array, B_array), ParamLibrary(mW_array, mB_array))
        self.converter.convert_network_format_to_MTNN(param_library, momentum_library)
        return param_library, momentum_library

    def insert_into_network(self, level, param_library, momentum_library):
        """Insert ParamLibrary tensors into a network.

        Inputs:
        level <Level>. The level into which to insert the parameters.
        net_param_library <ParamLibrary>. The tensors of network parameters to insert.
        momentum_library <ParamLibrary>. The tensors of momentum values to insert.

        Output: None.
        """
        # TODO: This function uses two parameter copies which is
        # unnecessary. Refactor to only use one.
        level.init_params = copy.deepcopy(param_library)
        level.init_momentum = copy.deepcopy(momentum_library)

        self.converter.convert_MTNN_format_to_network(param_library, momentum_library)
        W_array, B_array = param_library
        mW_array, mB_array = momentum_library

        # Insert parameters into the network
        with torch.no_grad():
            for layer_id in range(len(level.net.layers)):
                level.net.layers[layer_id].weight.copy_(W_array[layer_id])
                level.net.layers[layer_id].bias.copy_(B_array[layer_id].reshape(-1))
        level.net.zero_grad()

        # Insert momemntum values into the optimizer
        level.presmoother.momentum_data = []
        with torch.no_grad():
            for i in range(len(mW_array)):
                level.presmoother.momentum_data.append(mW_array[i].clone())
                level.presmoother.momentum_data.append(mB_array[i].clone().reshape(-1))
        level.presmoother.optimizer = None

    def add_to_network(self, level, param_diff_library, momentum_diff_library):
        """Add ParamLibrary difference value tensors to a network.

        This is similar to insert_into_network, except that it does not
        replace the values of the network but instead adds the inputs
        to the existing values. This is done, for example, in a Full
        Approximation Scheme coarse-grid correction.

        Inputs:
        level <Level>. The level into which to insert the parameters.
        net_param_library <ParamLibrary>. The tensors of network parameters to insert.
        momentum_library <ParamLibrary>. The tensors of momentum values to insert.

        Output: None.

        """
        self.converter.convert_MTNN_format_to_network(param_diff_library, momentum_diff_library)
        dW_array, dB_array = param_diff_library
        dmW_array, dmB_array = momentum_diff_library

        optimizer = level.presmoother.optimizer
        get_p = lambda ind : optimizer.state[optimizer.param_groups[0]['params'][ind]]['momentum_buffer']

        with torch.no_grad():
            for layer_id in range(len(level.net.layers)):
                # Add parameters to the network
                level.net.layers[layer_id].weight.add_(dW_array[layer_id])
                level.net.layers[layer_id].bias.add_(dB_array[layer_id].reshape(-1))
                # Add momentum values to the optimizer.
                get_p(2*layer_id).add_(dmW_array[layer_id])
                get_p(2*layer_id+1).add_(dmB_array[layer_id].reshape(-1))
    

############################################
# Converters
############################################

class SecondOrderConverter:
    """Converts parameter libraries between format.

    A ParamLibrary contains lists of weight and bias tensors. The
    format of these tensors as required by the neural networks is
    different than the format required by torch for its matrix
    multiplication broadcasting semantics, which we use during
    restriction and prolongation. This class converts between these
    two formats.

    """
    
    @abstractmethod
    def convert_network_format_to_MTNN(self, param_library, momentum_library):
        """ Convert tensors in a ParamLibrary from network format to MTNN in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        Input: <ParamLibrary>
        Output: None
        """
        raise NotImplementedError

    
    @abstractmethod
    def convert_MTNN_format_to_network(self, param_library, momentum_library):
        """ Convert tensors in a ParamLibrary from MTNN format to network format in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        Input: <ParamLibrary>
        Output: None
        """
        raise NotImplementedError
    

class MultiLinearConverter(SecondOrderConverter):
    """MultiLinearConverter

    A SecondOrderConverter for multilinear, aka fully-connected, models.
    """
    def convert_network_format_to_MTNN(self, param_library, momentum_library):
        pass

    def convert_MTNN_format_to_network(self, param_library, momentum_library):
        pass


class ConvolutionalConverter(SecondOrderConverter):
    """ConvolutionalConverter

    A SecondOrderConverter for convolutional networks that consist of
    1 or more convolutional layers followed by 0 or more
    fully-connected layers.
    """
    def __init__(self, num_conv_layers):
        self.num_conv_layers = num_conv_layers
    
    def convert_network_format_to_MTNN(self, param_library, momentum_library):
        # Reorder convolutional tensors
        for layer_id in range(self.num_conv_layers):
            # Has shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
            W = param_library.weights[layer_id]
            mW = momentum_library.weights[layer_id]
            # Convert to shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
            param_library.weights[layer_id] = W.permute(*range(2, len(W.shape)), 0, 1)
            momentum_library.weights[layer_id] = mW.permute(*range(2, len(W.shape)), 0, 1)

        # Each neuron in the first FC layer after convolutional layers
        # has a separate weight for each pixel and channel. For each
        # pixel, there is a set of columns in the FC weight matrix
        # associated with that pixel's channels, and we need to merge
        # those columns acoording to our coarsening. Thus, we need to
        # convert this weight matrix into a 3rd order tensor with
        # dimensions (# pixels, # neurons, # last_layer_channnels).
        layer_id = self.num_conv_layers
        last_layer_channels = param_library.weights[layer_id-1].shape[-2]
        W = param_library.weights[layer_id]
        mW = momentum_library.weights[layer_id]

        # Last conv layer output has shape (minibatches, out_ch,
        # pixel_rows, pixel_cols) and gets flattened to shape
        # (minibatches, outch * pixels_rows * pixel_cols), which keeps
        # last indexed elements together. That is, it's a sort of
        # "channel-major order."
        
        # Creates tensor of shape (neurons, last_layer_channels, # pixels)
        Wnew = W.reshape(W.shape[0], last_layer_channels, int(W.shape[1] / last_layer_channels))
        mWnew = mW.reshape(mW.shape[0], last_layer_channels, int(mW.shape[1] / last_layer_channels))
        # Permute to correct shape
        Wnew = Wnew.permute(2, 0, 1)
        mWnew = mWnew.permute(2, 0, 1)

        param_library.weights[layer_id] = Wnew
        momentum_library.weights[layer_id] = mWnew
    
    def convert_MTNN_format_to_network(self, param_library, momentum_library):
        # Reorder convolutional tensors
        for layer_id in range(self.num_conv_layers):
            # Has shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
            W = param_library.weights[layer_id]
            mW = momentum_library.weights[layer_id]
            # Convert to shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
            param_library.weights[layer_id] = W.permute(-2, -1, *range(len(W.shape)-2))
            momentum_library.weights[layer_id] = mW.permute(-2, -1, *range(len(W.shape)-2))

        # First FC layer has shape (# pixels, out_neurons,
        # last_layer_channels).  Needs to have shape (out_neurons,
        # all_input_values) and needs to maintain "channel-major"
        # ordering.
        layer_id = self.num_conv_layers
        W = param_library.weights[layer_id]
        W = torch.flatten(W.permute(1, 2, 0), 1)
        mW = momentum_library.weights[layer_id]
        mW = torch.flatten(mW.permute(1, 2, 0), 1)
        param_library.weights[layer_id] = W
        momentum_library.weights[layer_id] = mW

        
class ActivationDistanceConverter(SecondOrderConverter):
    """ActivationDistanceConverter

    A neuron's activation distance (for a ReLU activation function) is
    the distance from the origin at which the input crosses the zero
    threshold, allowing the neuron to "activate." That distance is
    given by the formula
    activation_distance = -bias / ||neuron_weights||

    This converter acts as a modifier around another primary
    converter. After the primary converter performs its conversion,
    this class will convert biases into activation distances. The
    reason to do this is if you would like to consider activation
    distance to be the true parameter that we coarsen instead of bias.
    """
    
    def __init__(self, primary_converter, half_for_average = True):
        """Constructor

        Inputs: 
        primary_converter (SecondOrderConverter) The primary
                          conversion method around which this wraps.
        half_for_average (bool) If true, cut activation distances in 
                          half so when they are summed we get an average.
        """
        self.primary_converter = primary_converter
        self.half_for_average = half_for_average

    def convert_biases_to_activation_distances(self, library):
        """ Convert biases in a ParamLibrary into activation distances.
        Place activation distance in what is normally the bias spot.

        Inputs:
        library (ParamLibrary)
        """
        for layer_id in range(len(library.weights)):
            W = library.weights[layer_id]
            W = W.view(W.shape).transpose(dim0=0, dim1=-2)
            W = W.reshape(W.shape[0], -1) # This call induces a copy, which is inefficient
            B = library.biases[layer_id]
            library.biases[layer_id] = -B / torch.norm(W, p=2, dim=1, keepdim=True)
            if self.half_for_average:
                library.biases[layer_id] /= 2.0

    def convert_network_format_to_MTNN(self, param_library, momentum_library):
        """ Perform conversion.
        First primary conversion. Then bias to activation distance.
        """
        self.primary_converter.convert_network_format_to_MTNN(param_library, momentum_library)
        self.convert_biases_to_activation_distances(param_library)
        self.convert_biases_to_activation_distances(momentum_library)

    def convert_activation_distances_to_biases(self, library):
        """ Convert activation distances in a ParamLibrary back into biases.

        Inputs:
        library (ParamLibrary)
        """
        for layer_id in range(len(library.weights)):
            W = library.weights[layer_id]
            W = W.view(W.shape).transpose(dim0=0, dim1=-2)
            W = W.reshape(W.shape[0], -1) # This call induces a copy, which is inefficient
            AD = library.biases[layer_id]
            library.biases[layer_id] = -AD * torch.norm(W, p=2, dim=1, keepdim=True)
            if self.half_for_average:
                library.biases[layer_id] *= 2.0

    def convert_MTNN_format_to_network(self, param_library, momentum_library):
        """ Perform conversion back.
        First activation distance back to bias. Then primary conversion.
        """
        self.convert_activation_distances_to_biases(param_library)
        self.convert_activation_distances_to_biases(momentum_library)
        self.primary_converter.convert_MTNN_format_to_network(param_library, momentum_library)
