"""
Restriction Operators
"""
import torch
from abc import ABC, abstractmethod

# local
import MTNN.utils.logger as log

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['SecondOrderConverter',
           'MultiLinearConverter',
           'ConvolutionalConverter']    

############################################
# Converters
############################################
class SecondOrderConverter:
    """Converts parameter libraries between format.

    We currently support "second order restrictions," in which we
    apply restriction and prolongation via matrix multiplication on
    either side of a tensor. That is, if a parameter tensor W has
    dimensions $(d_1, d_2, ..., d_k, m, n)$, then we restrict via
    $R_op @ W @ P_op$, which, for each choice of indices $(i_1, ...,
    i_k)$ over the first $k$ dimensions, performs matrix
    multiplicaiton over the last two dimensions. We think of this as
    "second order" because the restriction operation is quadratic in
    $(R_op, P_op)$.

    A ParamVector contains lists of weight and bias tensors. The
    format of these tensors as required by the neural networks is
    different than the format required by PyTorch for its matrix
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

        @param param_vector The ParamVector to convert to MTNN format.
        """
        raise NotImplementedError

    
    @abstractmethod
    def convert_MTNN_format_to_network(self, param_vector):
        """ Convert tensors in a ParamVector from MTNN format to network format in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        @param param_vector The ParamVector to convert to neural network format.
        """
        raise NotImplementedError
    

class MultiLinearConverter(SecondOrderConverter):
    """MultiLinearConverter

    A SecondOrderConverter for multilinear, aka fully-connected, models.

    In this case, NN and MTNN formats are the same, so conversion is
    trivial.
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

    For convolutional networks, our restriction operation merges
    channels together. It does not change the expected number of
    pixels in an input tensor or the kernel size. Thus, converting to
    MTNN format requires reordering the weight tensors so that PyTorch
    broadcasting semantics broadcast over kernel dimensions while
    summing over channels.

    This class has been tested for second order input tensors (ie
    images), but in theory it works for inputs tensors of any order.
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
        # those columns according to our coarsening. Thus, we need to
        # convert this weight matrix into a 3rd order tensor with
        # dimensions (# pixels, # neurons, # last_layer_channnels).
        layer_id = self.num_conv_layers
        last_layer_output_channels = param_vector.weights[layer_id-1].shape[-2] # num output channels
        W = param_vector.weights[layer_id]

        # Last conv layer output has shape (minibatch_size, out_ch,
        # pixel_rows, pixel_cols) and gets flattened to shape
        # (minibatches, out_ch * pixels_rows * pixel_cols), which
        # keeps last indexed elements together. That is, it's a sort
        # of "channel-major order" in that it keeps all the values
        # associated with a given channel together.
        
        # Creates tensor of shape (neurons, last_layer_output_channels, num_pixels)
        Wnew = W.reshape(W.shape[0], last_layer_output_channels,
                         int(W.shape[1] / last_layer_output_channels))
        # Permute to shape (num_pixels, num_neurons, last_layer_output_channels)
        Wnew = Wnew.permute(2, 0, 1)

        param_vector.weights[layer_id] = Wnew
    
    def convert_MTNN_format_to_network(self, param_vector):
        # Reorder convolutional tensors
        for layer_id in range(self.num_conv_layers):
            # Has shape (kernel_d1 x kernel_d2 x ... x kernel_dk x out_ch x in_ch)
            W = param_vector.weights[layer_id]
            # Convert to shape (out_ch x in_ch x kernel_d1 x kernel_d2 x ... x kernel_dk)
            param_vector.weights[layer_id] = W.permute(-2, -1, *range(len(W.shape)-2))

        # First FC layer has shape (# pixels, num_neurons,
        # last_layer_output_channels).  Needs to have shape
        # (num_neurons, all_input_values) and needs to maintain
        # "channel-major" ordering.
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

        @param primary_converter (SecondOrderConverter) The primary conversion method around which this wraps.
        @param half_for_average (bool) If true, cut activation distances in half so when they are summed we get an average.
        """
        self.primary_converter = primary_converter
        self.half_for_average = half_for_average

    def convert_biases_to_activation_distances(self, library):
        """ Convert biases in a ParamVector into activation distances.
        Place activation distance in what is normally the bias spot.

        Inputs:
        @param library (ParamVector)
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
