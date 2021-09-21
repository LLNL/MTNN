"""
Restriction Operators
"""
# local
import MTNN.utils.logger as log
import MTNN.utils.deviceloader as deviceloader

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['SecondOrderRestriction',
           'SecondOrderProlongation']


####################################################################
# Restriction
####################################################################

class SecondOrderRestriction:
    """This restriction class implements "second order restrictions" in
    the sense that we will apply restriction and prolongation via
    matrix multiplication on either side of a tensor. That is, if a
    parameter tensor W has dimensions (d1, d_2, ..., d_k, m, n), then
    we will restrict via R_op @ W @ P_op, which, for each choice of
    indices in the first k dimensions, performs matrix multiplication
    over the last two dimension.

    This can be thought of as a second order restriction because the
    restricted tensor is quadratic in the R_op and P_op matrices.

    """

    def __init__(self, parameter_extractor, matching_method, transfer_operator_builder, adjust_bias = False):
        """Construct the SecondOrderRestrcition.

        Inputs: 
        parameter_extractor <ParameterExtractor>

        matching_method <Callable>. Takes as input a ParamVector and
        a Level, produces as output a CoarseMapping object.

        transfer_operator_builder <Callable>. Takes as input a
        ParamVector, a CoarseMapping, and a torch.device, and
        produces a TransferOps object.

        adjust_bias <bool>. Whether or not to adjust the coarse biases
        by cos(theta/2), where theta is the weight vector angle
        between the two matched neurons.

        """
        self.parameter_extractor = parameter_extractor
        self.matching_method = matching_method
        self.transfer_operator_builder = transfer_operator_builder
        self.adjust_bias = adjust_bias

        self.coarse_mapping = None
        self.redo_matching_frequency = 10
        self.cycles_since_last_matching = self.redo_matching_frequency

    def apply(self, fine_level, coarse_level, dataloader, verbose=False):
        fine_param_library, fine_momentum_library = self.parameter_extractor.extract_from_network(fine_level)
        if self.cycles_since_last_matching >= self.redo_matching_frequency:
            self.coarse_mapping = self.matching_method(fine_param_library, fine_level.net)
            self.cycles_since_last_matching = 1
        else:
            self.cycles_since_last_matching += 1

        self.transfer_ops, self.tau_transfer_ops = self.transfer_operator_builder(fine_param_library, 
                                                                                  self.coarse_mapping,
                                                                                  deviceloader.get_device())
        coarse_param_library = self.transfer_ops @ fine_param_library
        coarse_momentum_library = self.transfer_ops @ fine_momentum_library

        # So this is some terrible software design right here. :-D
        # TODO: Refactor for architecture extensibility.
        if fine_level.net.__class__.__name__ == "MultiLinearNet":
            coarse_level_dims = [fine_level.net.layers[0].in_features] + self.coarse_mapping.num_coarse_channels + \
                [fine_level.net.layers[-1].out_features]
            coarse_level.net = fine_level.net.__class__(coarse_level_dims,
                                                        fine_level.net.activation,
                                                        fine_level.net.output)
            coarse_level.net.set_device(fine_level.net.device)
        elif fine_level.net.__class__.__name__ == "ConvolutionalNet":
            num_conv_layers = fine_level.net.num_conv_layers
            out_ch, kernel_widths, strides = zip(*fine_level.net.conv_channels)
            out_ch = [out_ch[0]] + self.coarse_mapping.num_coarse_channels[:num_conv_layers]
            conv_channels = list(zip(out_ch, kernel_widths, strides))
            first_fc_width = conv_channels[-1][0] * coarse_param_library.weights[num_conv_layers].shape[0]
            fc_dims = [first_fc_width] + self.coarse_mapping.num_coarse_channels[num_conv_layers:] + \
                      [fine_level.net.layers[-1].out_features]
            coarse_level.net = fine_level.net.__class__(conv_channels,
                                                        fc_dims,
                                                        fine_level.net.activation,
                                                        fine_level.net.output_activation)
        else:
            raise RuntimeError("SecondOrderRestriction::apply: {} is not a supported network type.".format(fine_level.net.__class__.__name__))
        
        self.parameter_extractor.insert_into_network(coarse_level, coarse_param_library,
                                                     coarse_momentum_library)

        coarse_level.corrector.compute_tau(coarse_level, fine_level, dataloader, self.tau_transfer_ops)

####################################################################
# Prolongation
####################################################################

class SecondOrderProlongation:
    """
    This class implement the inverse operation of SecondOrderRestriction.
    """
    
    def __init__(self, parameter_extractor, restriction):
        """Construct a SecondOrderProlongation.

        Inputs:
        parameter_extractor <ParameterExtractor>.
        restriction <SecondOrderRestriction>. The restriction object
        to which this SecondOrderProlongation is paired.
        """
        self.parameter_extractor = parameter_extractor
        self.restriction = restriction
        self.adjust_bias = self.restriction.adjust_bias

    def apply(self, fine_level, coarse_level, dataloader, verbose):
        assert(fine_level.id < coarse_level.id)

        coarse_param_library, coarse_momentum_library = self.parameter_extractor.extract_from_network(coarse_level)
        prolongation_ops = self.restriction.transfer_ops.swap_transfer_ops()

        coarse_param_diff_library = coarse_param_library - coarse_level.init_params
        coarse_momentum_diff_library = coarse_momentum_library - coarse_level.init_momentum

        fine_param_diff_library = prolongation_ops @ coarse_param_diff_library
        fine_momentum_diff_library = prolongation_ops @ coarse_momentum_diff_library

        self.parameter_extractor.add_to_network(fine_level, fine_param_diff_library,
                                                fine_momentum_diff_library)