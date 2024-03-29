"""
Restriction Operators
"""
# local
from MTNN.utils import logger
import MTNN.utils.deviceloader as deviceloader

__all__ = ['SecondOrderRestriction',
           'SecondOrderProlongation']


####################################################################
# Restriction
####################################################################

class SecondOrderRestriction:
    """This restriction class implements "second order restrictions" in
    the sense that we will apply restriction and prolongation via
    matrix multiplication on either side of a tensor. That is, if a
    parameter tensor W has dimensions $(d1, d_2, ..., d_k, m, n)$, then
    we will restrict via $R_op @ W @ P_op$, which, for each choice of
    indices in the first $k$ dimensions, performs matrix multiplication
    over the last two dimensions.

    This can be thought of as a second order restriction because the
    restriction operator is quadratic in $(R_op, P_op)$.

    Note that restriction operations of other orders are possible as
    well.

    """

    def __init__(self, parameter_extractor, matching_method, transfer_operator_builder, 
                 coarse_model_factory, redo_matching_frequency = 10, adjust_bias = False):
        """Construct the SecondOrderRestrcition.

        @param parameter_extractor <ParameterExtractor>

        @param matching_method <Callable>. Takes as input a
        ParamVector and a Level, produces as output a CoarseMapping
        object.

        @param transfer_operator_builder <Callable>. Takes as input a
        ParamVector, a CoarseMapping, and a torch.device, and produces
        a TransferOps object which can perform restriction and
        prolongation.

        @param coarse_model_factory <CoarseModelFactory>. Takes as
        input a fine neural network and a CoarseMapping and produces
        an uninitialized coarse network that has the appropriate
        architecture.

        @param redo_matching_frequency <int>. Redo the fine-to-coarse
        mapping every redo_matching_frequency cycles.

        @param adjust_bias <bool>. Whether or not to adjust the coarse
        biases by cos(theta/2), where theta is the weight vector angle
        between the two matched neurons.

        """
        self.parameter_extractor = parameter_extractor
        self.matching_method = matching_method
        self.transfer_operator_builder = transfer_operator_builder
        self.coarse_model_factory = coarse_model_factory
        self.adjust_bias = adjust_bias #deprecated? this looks not used

        self.coarse_mapping = None
        self.redo_matching_frequency = redo_matching_frequency
        self.cycles_since_last_matching = self.redo_matching_frequency

    def apply(self, fine_level, coarse_level, dataloader, verbose=False):
        fine_param_library, fine_momentum_library = self.parameter_extractor.extract_from_network(fine_level)

        # Periodically recompute coarse matching.
        if self.cycles_since_last_matching >= self.redo_matching_frequency:
            self.coarse_mapping = self.matching_method(fine_param_library, fine_level.net)
            self.cycles_since_last_matching = 1
        else:
            self.cycles_since_last_matching += 1

        # Build operators to compute coarse neural network parameters
        # and coarse tau correction.
        self.transfer_ops, self.tau_transfer_ops = self.transfer_operator_builder(fine_param_library, 
                                                                                  self.coarse_mapping,
                                                                                  deviceloader.get_device())

        # To enable multilevel momentum, we coarse both the neural
        # network and the momentum. The momentum values have the same
        # structure as the NN parameters, so we can apply the same
        # restriction operation.
        coarse_param_library = self.transfer_ops @ fine_param_library
        coarse_momentum_library = self.transfer_ops @ fine_momentum_library

        # Construct new NN that has the same architecture as the fine
        # NN but with coarsened parametrs.
        coarse_level.net = self.coarse_model_factory.build(fine_level.net, self.coarse_mapping)

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
    
    def __init__(self, parameter_extractor, restriction, param_diff_scale = 1.0, mom_diff_scale = 1.0):
        """Construct a SecondOrderProlongation.

        @param parameter_extractor <ParameterExtractor>.

        @param restriction <SecondOrderRestriction>. The restriction
        object to which this SecondOrderProlongation is paired.

        @param param_diff_scale Upon prolongation, scale the parameter
        correction by this amount before adding it to the fine
        parameters.

        @param mom_diff_scale Upon prolongation, scale the momentum
        correction by this amount before adding it to the fine
        momentum.

        """
        self.parameter_extractor = parameter_extractor
        self.restriction = restriction
        self.adjust_bias = self.restriction.adjust_bias
        self.param_diff_scale = param_diff_scale
        self.mom_diff_scale = mom_diff_scale

    def apply(self, fine_level, coarse_level, dataloader, verbose):
        assert(fine_level.id < coarse_level.id)

        coarse_param_library, coarse_momentum_library = self.parameter_extractor.extract_from_network(coarse_level)

        # If restriction is computed as $W_C = R_op @ W @ P_op$, then
        # prolongation is computed as $W = P_op @ W_c @ R_op$, so just
        # swap operators.
        prolongation_ops = self.restriction.transfer_ops.swap_transfer_ops()

        # In the Full Approximation Scheme, $W \leftarrow W + P(W_c -
        # RW), so need to compute difference with initially-coarened
        # values and then prolong that.
        coarse_param_diff_library = coarse_param_library - coarse_level.init_params
        coarse_momentum_diff_library = coarse_momentum_library - coarse_level.init_momentum

        fine_param_diff_library = self.param_diff_scale * (prolongation_ops @ coarse_param_diff_library)
        fine_momentum_diff_library = self.mom_diff_scale * (prolongation_ops @ coarse_momentum_diff_library)

        self.parameter_extractor.add_to_network(fine_level, fine_param_diff_library,
                                                fine_momentum_diff_library)
