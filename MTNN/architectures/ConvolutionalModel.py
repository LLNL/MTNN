import torch
from MTNN.architectures.architecture_interfaces import *

class ConvolutionalNet(BaseModel):
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

    def save_params(self, path):
        torch.save(self.layers, path)

    def load_params(self, path):
        self.layers = torch.load(path)

    def forward(self, x):
        for i in range(self.num_conv_layers):
            x = self.layers[i](x)
            x = self.activation(x)

        x = torch.flatten(x, 1)

        for i in range(self.num_conv_layers, len(self.layers)-1):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.layers[-1](x)
        x = self.output_activation(x)

        return x

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


class CoarseConvolutionalFactory(CoarseModelFactory):
    def get_num_pixels_in_last_conv_layer(self, fine_network):
        # The # of columns in the weight matrix of the first FC layer
        # is #channels x #pixels from the last convolutional layer, so
        # compute #columns / #channels to get # pixels.
        num_conv_layers = fine_network.num_conv_layers
        num_last_conv_out_ch = fine_network.layers[num_conv_layers-1].weight.shape[0]
        num_pixels = int(fine_network.layers[num_conv_layers].weight.shape[1] / num_last_conv_out_ch)
        return num_pixels

    def build(self, fine_network, coarse_mapping):
        num_conv_layers = fine_network.num_conv_layers
        out_ch, kernel_widths, strides = zip(*fine_network.conv_channels)
        out_ch = [out_ch[0]] + coarse_mapping.num_coarse_channels[:num_conv_layers]
        conv_channels = list(zip(out_ch, kernel_widths, strides))
        first_fc_width = conv_channels[-1][0] * self.get_num_pixels_in_last_conv_layer(fine_network)
        fc_dims = [first_fc_width] + coarse_mapping.num_coarse_channels[num_conv_layers:] + \
                  [fine_network.layers[-1].out_features]

        coarse_net = fine_network.__class__(conv_channels, fc_dims,
                                            fine_network.activation,
                                            fine_network.output_activation)
        coarse_net.set_device(fine_network.device)
        return coarse_net



