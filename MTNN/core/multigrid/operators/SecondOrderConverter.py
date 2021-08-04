"""
Restriction Operators
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from abc import ABC, abstractmethod

# local
import MTNN.core.multigrid.scheme as mg
import MTNN.core.components.models as models
from MTNN.utils.datatypes import operators, ParamVector
import MTNN.utils.logger as log
import MTNN.utils.printer as printer
import MTNN.utils.deviceloader as deviceloader

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['SecondOrderConverter',
           'MultilinearConverter',
           'ConvolutionalConverter']    

############################################
# Converters
############################################

class SecondOrderConverter:
    """Converts parameter libraries between format.

    A ParamVector contains lists of weight and bias tensors. The
    format of these tensors as required by the neural networks is
    different than the format required by torch for its matrix
    multiplication broadcasting semantics, which we use during
    restriction and prolongation. This class converts between these
    two formats.

    """
    
    @abstractmethod
    def convert_network_format_to_MTNN(self, param_vector):
        """ Convert tensors in a ParamVector from network format to MTNN in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        Input: <ParamVector>
        Output: None
        """
        raise NotImplementedError

    
    @abstractmethod
    def convert_MTNN_format_to_network(self, param_vector):
        """ Convert tensors in a ParamVector from MTNN format to network format in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        Input: <ParamVector>
        Output: None
        """
        raise NotImplementedError
    

class MultiLinearConverter(SecondOrderConverter):
    """MultiLinearConverter

    A SecondOrderConverter for multilinear, aka fully-connected, models.
    """
    def convert_network_format_to_MTNN(self, param_vector):
        pass

    def convert_MTNN_format_to_network(self, param_vector):
        pass


class ConvolutionalConverter(SecondOrderConverter):
    """ConvolutionalConverter

    A SecondOrderConverter for convolutional networks that consist of
    1 or more convolutional layers followed by 0 or more
    fully-connected layers.
    """
    def __init__(self, num_conv_layers):
        self.num_conv_layers = num_conv_layers
    
    def convert_network_format_to_MTNN(self, param_vector):
        # Reorder convolutional tensors
        for layer_id in range(self.num_conv_layers):
            # Has shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
            W = param_vector.weights[layer_id]
            # Convert to shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
            param_vector.weights[layer_id] = W.permute(*range(2, len(W.shape)), 0, 1)

        # Each neuron in the first FC layer after convolutional layers
        # has a separate weight for each pixel and channel. For each
        # pixel, there is a set of columns in the FC weight matrix
        # associated with that pixel's channels, and we need to merge
        # those columns acoording to our coarsening. Thus, we need to
        # convert this weight matrix into a 3rd order tensor with
        # dimensions (# pixels, # neurons, # last_layer_channnels).
        layer_id = self.num_conv_layers
        last_layer_channels = param_vector.weights[layer_id-1].shape[-2]
        W = param_vector.weights[layer_id]

        # Last conv layer output has shape (minibatches, out_ch,
        # pixel_rows, pixel_cols) and gets flattened to shape
        # (minibatches, outch * pixels_rows * pixel_cols), which keeps
        # last indexed elements together. That is, it's a sort of
        # "channel-major order."
        
        # Creates tensor of shape (neurons, last_layer_channels, # pixels)
        Wnew = W.reshape(W.shape[0], last_layer_channels, int(W.shape[1] / last_layer_channels))
        # Permute to correct shape
        Wnew = Wnew.permute(2, 0, 1)

        param_vector.weights[layer_id] = Wnew
    
    def convert_MTNN_format_to_network(self, param_vector):
        # Reorder convolutional tensors
        for layer_id in range(self.num_conv_layers):
            # Has shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
            W = param_vector.weights[layer_id]
            # Convert to shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
            param_vector.weights[layer_id] = W.permute(-2, -1, *range(len(W.shape)-2))

        # First FC layer has shape (# pixels, out_neurons,
        # last_layer_channels).  Needs to have shape (out_neurons,
        # all_input_values) and needs to maintain "channel-major"
        # ordering.
        layer_id = self.num_conv_layers
        W = param_vector.weights[layer_id]
        W = torch.flatten(W.permute(1, 2, 0), 1)
        param_vector.weights[layer_id] = W

        
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
        """ Convert biases in a ParamVector into activation distances.
        Place activation distance in what is normally the bias spot.

        Inputs:
        library (ParamVector)
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
        """ Convert activation distances in a ParamVector back into biases.

        Inputs:
        library (ParamVector)
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




















# class ConvolutionalConverter(SecondOrderConverter):
#     """ConvolutionalConverter

#     A SecondOrderConverter for convolutional networks that consist of
#     1 or more convolutional layers followed by 0 or more
#     fully-connected layers.
#     """
#     def __init__(self, num_conv_layers):
#         self.num_conv_layers = num_conv_layers
    
#     def convert_network_format_to_MTNN(self, param_library, momentum_library):
#         # Reorder convolutional tensors
#         for layer_id in range(self.num_conv_layers):
#             # Has shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
#             W = param_library.weights[layer_id]
#             mW = momentum_library.weights[layer_id]
#             # Convert to shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
#             param_library.weights[layer_id] = W.permute(*range(2, len(W.shape)), 0, 1)
#             momentum_library.weights[layer_id] = mW.permute(*range(2, len(W.shape)), 0, 1)

#         # Each neuron in the first FC layer after convolutional layers
#         # has a separate weight for each pixel and channel. For each
#         # pixel, there is a set of columns in the FC weight matrix
#         # associated with that pixel's channels, and we need to merge
#         # those columns acoording to our coarsening. Thus, we need to
#         # convert this weight matrix into a 3rd order tensor with
#         # dimensions (# pixels, # neurons, # last_layer_channnels).
#         layer_id = self.num_conv_layers
#         last_layer_channels = param_library.weights[layer_id-1].shape[-2]
#         W = param_library.weights[layer_id]
#         mW = momentum_library.weights[layer_id]

#         # Last conv layer output has shape (minibatches, out_ch,
#         # pixel_rows, pixel_cols) and gets flattened to shape
#         # (minibatches, outch * pixels_rows * pixel_cols), which keeps
#         # last indexed elements together. That is, it's a sort of
#         # "channel-major order."
        
#         # Creates tensor of shape (neurons, last_layer_channels, # pixels)
#         Wnew = W.reshape(W.shape[0], last_layer_channels, int(W.shape[1] / last_layer_channels))
#         mWnew = mW.reshape(mW.shape[0], last_layer_channels, int(mW.shape[1] / last_layer_channels))
#         # Permute to correct shape
#         Wnew = Wnew.permute(2, 0, 1)
#         mWnew = mWnew.permute(2, 0, 1)

#         param_library.weights[layer_id] = Wnew
#         momentum_library.weights[layer_id] = mWnew
    
#     def convert_MTNN_format_to_network(self, param_library, momentum_library):
#         # Reorder convolutional tensors
#         for layer_id in range(self.num_conv_layers):
#             # Has shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
#             W = param_library.weights[layer_id]
#             mW = momentum_library.weights[layer_id]
#             # Convert to shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
#             param_library.weights[layer_id] = W.permute(-2, -1, *range(len(W.shape)-2))
#             momentum_library.weights[layer_id] = mW.permute(-2, -1, *range(len(W.shape)-2))

#         # First FC layer has shape (# pixels, out_neurons,
#         # last_layer_channels).  Needs to have shape (out_neurons,
#         # all_input_values) and needs to maintain "channel-major"
#         # ordering.
#         layer_id = self.num_conv_layers
#         W = param_library.weights[layer_id]
#         W = torch.flatten(W.permute(1, 2, 0), 1)
#         mW = momentum_library.weights[layer_id]
#         mW = torch.flatten(mW.permute(1, 2, 0), 1)
#         param_library.weights[layer_id] = W
#         momentum_library.weights[layer_id] = mW
