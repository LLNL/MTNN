from MTNN.utils.datatypes import CoarseMapping
from MTNN.utils import logger
import MTNN.utils.deviceloader as deviceloader

class CoarseModelFactory:
    """Constructs coarse neural networks based on fine networks and coarse mapping information."""
    def build(self, fine_network, coarse_mapping):
        raise NotImplementedError

class CoarseMultilinearFactory(CoarseModelFactory):
    def build(self, fine_network, coarse_mapping):
        print("In coarse multilinear factory.")
        coarse_level_dims = [fine_network.layers[0].in_features] + coarse_mapping.num_coarse_channels + \
                            [fine_network.layers[-1].out_features]
        coarse_net = fine_network.__class__(coarse_level_dims,
                                              fine_network.activation,
                                              fine_network.output)
        coarse_net.set_device(fine_network.device)
        return coarse_net

class CoarseConvolutionalFactory(CoarseModelFactory):
    def build(self, fine_network, coarse_mapping):
        num_conv_layers = fine_network.num_conv_layers
        out_ch, kernel_widths, strides = zip(*fine_network.conv_channels)
        out_ch = [out_ch[0]] + coarse_mapping.num_coarse_channels[:num_conv_layers]
        conv_channels = list(zip(out_ch, kernel_widths, strides))
        first_fc_width = conv_channels[-1][0] * coarse_param_library.weights[num_conv_layers].shape[0]
        fc_dims = [first_fc_width] + coarse_mapping.num_coarse_channels[num_conv_layers:] + \
                  [fine_network.layers[-1].out_features]
        coarse_net = fine_network.__class__(conv_channels,
                                              fc_dims,
                                              fine_network.activation,
                                              fine_network.output_activation)
        coarse_net.set_device(fine_network.device)
        return coarse_net


