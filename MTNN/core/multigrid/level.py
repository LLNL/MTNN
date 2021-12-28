from collections import namedtuple

import torch.nn as nn

import MTNN.utils.logger as log
import MTNN.utils.printer as printer
import MTNN.utils.datatypes as mgdata
from MTNN.core.multigrid.operators import taucorrector, smoother
import MTNN.core.multigrid.operators.second_order_transfer as SOR
import MTNN.core.multigrid.operators.data_converter as SOC
import MTNN.core.multigrid.operators.paramextractor as PE
import MTNN.core.multigrid.operators.similarity_matcher as SimilarityMatcher
import MTNN.core.multigrid.operators.transfer_ops_builder as TransferOpsBuilder


log = log.get_logger(__name__, write_to_file =True)

class Level:
    """A Level in a multilevel hierarchy contains a neural network as well
       as transfer operators between it and the next level.

    """
    def __init__(self, id: int, net, smoother, prolongation, restriction,
                 num_smoothing_passes, corrector=None):
        """
        @param id level id (assumed to be unique)
        @param net A neural network
        @param smoother smoothing method
        @param prolongation prolongation method
        @param restriction restriction method
        @param num_smoothing_passes number of passes to execute through current dataloader at each smoothing step
        @param corrector tau corrector, used for computing FAS tau correction
        """
        self.net = net
        self.id = id
        self.smoother = smoother
        self.prolongation = prolongation
        self.restriction = restriction
        self.corrector = corrector
        self.num_smoothing_passes =  num_smoothing_passes
        self.l2_info = None

        # Data attributes
        # TODO: tokenize?
        self.interpolation_data = None
        # Lhs
        self.Winit = None
        self.Binit = None

    def presmooth(self, model, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'PRESMOOTHING {self.smoother.__class__.__name__}',))
            self.smoother.apply(model, dataloader, self.num_smoothing_passes, tau=self.corrector,
                                   l2_info = self.l2_info, verbose=verbose)
        except Exception:
            raise

    def postsmooth(self, model, dataloader , verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'POSTSMOOTHING {self.smoother.__class__.__name__}'))
            self.smoother.apply(model, dataloader, self.num_smoothing_passes, tau=self.corrector,
                                    l2_info = self.l2_info, verbose=verbose)
        except Exception:
           raise

    def coarse_solve(self, model, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'COARSEST SMOOTHING {self.smoother.__class__.__name__}', border="*"))
            self.smoother.apply(model, dataloader, self.num_smoothing_passes, tau=self.corrector,
                                l2_info = self.l2_info,verbose=verbose)
        except Exception:
            raise

    def prolong(self, fine_level, coarse_level, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'PROLONGATING {self.prolongation.__class__.__name__}'))

            self.prolongation.apply(fine_level, coarse_level, dataloader, verbose)
        except Exception:
            raise

    def restrict(self, fine_level, coarse_level, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'RESTRICTING {self.restriction.__class__.__name__}'))
            self.restriction.apply(fine_level, coarse_level,  dataloader,  verbose)
        except Exception:
            raise

    def view(self):
        """Logs level attributes"""
        for atr in self.__dict__:
            atrval = self.__dict__[atr]
            if type(atrval) in (int, float, str, list, bool):
                log.info(f"\t{atr}: \t{atrval} ")
            elif isinstance(atrval, mgdata.operators):
                log.info(f"\t{atr}: \n\t\tRestriction: {atrval.R_op} "
                         f"\n\t\tProlongation: {atrval.P_op}")
            elif isinstance(atrval, mgdata.rhs):
                log.info(f"\t{atr}: {atrval}")
            else:
                log.info(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__}")



class HierarchyBuilder:
    """Builds a multilevel hierarchy of neural networks.

    The MTNN software is designed so that the user can make different
    choices for different components of the multilevel framework and
    MTNN can simply drop those choices in the right place. This
    builder allows the user to set those choices and then builds the
    framework properly for her.

    Uses the builder software design pattern so you can procedurally
    build up the hierarchy you want. Call the set_*
    functions to specify the components, then call build_hierarchy to
    create the multilevel hierarchy (a list of Level objects).

    """

    #==================================
    # Required
    #==================================
    
    def __init__(self, num_levels):
        """ Create HierarchyBuilder

        @param num_levels number of levels in the NN hierarchy
        """
        self.num_levels = num_levels
        self.num_smoothing_passes = 1

    def set_loss(self, loss_t):
        """Loss function to use for training."""
        self.loss_t = loss_t
        return self
    
    def set_stepper_params(self, learning_rate, momentum, weight_decay):
        """Standard SGD optimizer parameters used during smoothing."""
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        return self
    
    def set_smoother(self, smoother_t):
        """Set the smoother (ie one-level optimizer) to use.

        @param smoother_t <Class> The smoother data type.
        """
        self.smoother_t = smoother_t
        return self
    
    def set_converter(self, converter_t):
        """Set the NN-to-MTNN format converter, which is used in the
        ParameterExtractor to convert back and forth between tensor
        formats appropriate for NN application and formats appopriate
        for restricton/prolongation.

        Currently supports MultiLinearConverter and
        ConvolutionalConverter. See SecondOrderConverter and its
        subclasses for details.

        @param converter_t <Callable> Function which constructs a converter.

        """
        self.converter_t = converter_t
        return self

    def set_extractors(self, parameter_extractor_t, gradient_extractor_t):
        """Set the parameter and gradient extractor methods, which are used to
        pull relevant tensors from a NN for restriction/prolongation.

        Currently supports ParamMomentumExtractor and
        GradientExtractor which are appropriate for networks having a
        sequence of layers each of which contains a weight and bias
        tensor.

        @param parameter_extractor_t <Callable> Function which takes
        as input a converter object and constructs a
        parameter_extractor.

        @param gradient_extractor_t <Callable> Function which takes as
        input a converter object and constructs a gradient_extractor.

        """
        self.parameter_extractor_t = parameter_extractor_t
        self.gradient_extractor_t = gradient_extractor_t
        return self
    
    def set_matching_method(self, matching_method_t):
        """Set the matching method.

        Currently supports HEMMatcher and a few different similarity
        calculators.

        @param matching_method_t <Callable> Function which constructs
        a matching method.

        """
        self.matching_method_t = matching_method_t
        return self
    
    def set_transfer_ops_builder(self, transfer_ops_builder_t):
        """Set the transfer operator building method.

        Currently supports PairwiseOpsBuilder and the more efficient
        matrix-free version, PairwiseOpsBuilder_MatrixFree.

        @param transfer_ops_builder_t Function which constructs a
        transfer operator builder.

        """
        self.transfer_ops_builder_t = transfer_ops_builder_t
        return self
    
    def set_restriction_prolongation(self, restriction_t, prolongation_t):
        """Set the restriction and prolongation methods.

        Currently supports SecondOrderRestriction and
        SecondOrderProlongation. See those classes for details.

        @param restriction_t <Class> The restriction operator data type.
        @param prolongation_t <class> The prolongation operator data type.

        """
        self.restriction_t = restriction_t
        self.prolongation_t = prolongation_t
        return self
    
    def set_tau_corrector(self, tau_t):
        """Set the tau correction method.

        If $w$ is the unrolled vector of all learnable parameters in
        a model and $\tau$ is a vector of the same length, the tau
        corrector adds a term of the form $-w^T \tau$ to the objective
        function. The tau corrector, not used at the finest level,
        alters the coarse gradient so that, immediately after
        restriction, it is a restricted analogue of the fine
        gradient. This enables the coarse model to "learn like" the
        fine level, at least for the first few iterations.

        Currently supports
        NullTau - always sets tau to 0.
        WholeSetTau - computes tau vector over the whole set given this V-cycle and keeps it constant until the next V-cycle
        MinibatchTau - computes a separate tau vector for each minibatch

        @param tau_t <Callable> A function which takes as input a loss
        function and a gradient extractor and produces a tau
        correction method.

        """
        self.tau_t = tau_t

    def set_neural_network(self, net):
        """Set the finest-level neural network in the hierarchy."""
        self.net = net
        return self

    #=====================================
    # Optional
    #=====================================

    def set_num_smoothing_passes(self, num_smoothing_passes):
        """Set number of passes through current dataloader at each smoothing
        step. Optional, HierarchyBuilder uses default value of 1.

        @param num_smoothing_passes Number of passes
        """
        self.num_smoothing_passes = num_smoothing_passes
        return self

    #=====================================
    # Construction functionality
    #=====================================

    def build_hierarchy(self):
        hierarchy_levels = []
        for level_ind in range(self.num_levels):
            loss_fn = self.loss_t()
            sgd_smoother = self.smoother_t(loss_fn, self.learning_rate, self.momentum,
                                           self.weight_decay, log_interval = 1)
            converter = self.converter_t()
            parameter_extractor = self.parameter_extractor_t(converter)
            gradient_extractor = self.gradient_extractor_t(converter)
            matching_method = self.matching_method_t()
            transfer_operator_builder = self.transfer_ops_builder_t()
            restriction = self.restriction_t(parameter_extractor, matching_method, transfer_operator_builder)
            prolongation = self.prolongation_t(parameter_extractor, restriction)

            curr_level = Level(id = level_ind,
                               net = self.net,
                               smoother = sgd_smoother,
                               prolongation = prolongation,
                               restriction = restriction,
                               num_smoothing_passes = self.num_smoothing_passes,
                               corrector = self.tau_t(loss_fn, gradient_extractor))
                               
            hierarchy_levels.append(curr_level)
        return hierarchy_levels

    @classmethod
    def build_standard_from_params(cls, net, params):
        """Build a hierarchy from a set of input parameters using "typical"
        choices for function approximation.

        Building a hierarchy for classification will be similar,
        though the loss function will need to be different.

        """
        levelbuilder = cls(params["num_levels"])
        levelbuilder.set_loss(nn.MSELoss)
        levelbuilder.set_stepper_params(params["learning_rate"], params["momentum"], params["weight_decay"])
        levelbuilder.set_smoother(smoother.SGDSmoother)

        nn_is_cnn = "conv_ch" in params
        if nn_is_cnn:
            levelbuilder.set_converter(lambda : SOC.ConvolutionalConverter(net.num_conv_layers))
        else:
            levelbuilder.set_converter(SOC.MultiLinearConverter)
        
        levelbuilder.set_extractors(PE.ParamMomentumExtractor, PE.GradientExtractor)
        levelbuilder.set_matching_method(
            lambda : SimilarityMatcher.HEMMatcher(similarity_calculator=SimilarityMatcher.StandardSimilarity(),
                                                  coarsen_on_layer=None))
        levelbuilder.set_transfer_ops_builder(
            lambda : TransferOpsBuilder.PairwiseOpsBuilder_MatrixFree(weighted_projection=params["weighted_projection"]))

        levelbuilder.set_restriction_prolongation(SOR.SecondOrderRestriction, SOR.SecondOrderProlongation)

        if params["tau_corrector"] == "null":
            levelbuilder.set_tau_corrector(taucorrector.NullTau)
        elif params["tau_corrector"] == "wholeset":
            levelbuilder.set_tau_corrector(taucorrector.WholeSetTau)
        elif params["tau_corrector"] == "minibatch":
            levelbuilder.set_tau_corrector(taucorrector.MinibatchTau)
        else:
            raise RuntimeError("Tau corrector '{}' does not exist.".format(params["tau_corrector"]))

        levelbuilder.set_neural_network(net)

        neural_net_levels = levelbuilder.build_hierarchy()
        return neural_net_levels
