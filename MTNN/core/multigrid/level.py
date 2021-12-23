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
    def __init__(self, id: int, presmoother, postsmoother, prolongation, restriction,
                 coarsegrid_solver, num_epochs, corrector=None):
        """
        Args:
            id: <int> level id (assumed to be unique)
            model:  <core.components.model> Model
            presmoother:  <core.alg.multigrid.operators.smoother> Smoother
            postsmoother: <core.alg.multigrid.operators.smoother> Smoother
            prolongation: <core.alg.multigrid.operators.prolongation> Prolongation
            restriction: <core.alg.multigrid.operators.restriction> Restriction
            coarsegrid_solver:  <core.alg.multigrid.operators.smoother> Smoother
            corrector: <core.multigrid.operators.tau_corrector> TauCorrector
        """
        self.net = None
        self.id = id
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarsegrid_solver = coarsegrid_solver
        self.prolongation = prolongation
        self.restriction = restriction
        self.corrector = corrector
        self.num_epochs =  num_epochs
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
                log.info(printer.format_header(f'PRESMOOTHING {self.presmoother.__class__.__name__}',))
            self.presmoother.apply(model, dataloader, self.num_epochs, tau=self.corrector,
                                   l2_info = self.l2_info, verbose=verbose)
        except Exception:
            raise

    def postsmooth(self, model, dataloader , verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'POSTSMOOTHING {self.postsmoother.__class__.__name__}'))
            self.postsmoother.apply(model, dataloader, self.num_epochs, tau=self.corrector,
                                    l2_info = self.l2_info, verbose=verbose)
        except Exception:
           raise

    def coarse_solve(self, model, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'COARSE SOLVING {self.coarsegrid_solver.__class__.__name__}', border="*"))
            self.coarsegrid_solver.apply(model, dataloader, self.num_epochs, tau=self.corrector,
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



class LevelBuilder:
    def __init__(self, num_levels):
        self.num_levels = num_levels
        self.num_epochs = 1

    def add_loss(self, loss_t):
        self.loss_t = loss_t
        return self
    def add_stepper_params(self, learning_rate, momentum, weight_decay):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        return self
    def add_smoother(self, smoother_t):
        self.smoother_t = smoother_t
        self.coarse_solver_t = smoother_t
        return self
    def add_coarse_level_solver(self, coarse_solver_t):
        self.coarse_solver_t = coarse_solver_t
        return self

    def set_epochs_per_cycle(self, num_epochs):
        self.num_epochs = num_epochs
        return self
        
    def add_converter(self, converter_t):
        self.converter_t = converter_t
        return self
    def add_extractors(self, parameter_extractor_t, gradient_extractor_t):
        self.parameter_extractor_t = parameter_extractor_t
        self.gradient_extractor_t = gradient_extractor_t
        return self
    def add_matching_method(self, matching_method_t):
        self.matching_method_t = matching_method_t
        return self
    def add_transfer_ops_builder(self, transfer_ops_builder_t):
        self.transfer_ops_builder_t = transfer_ops_builder_t
        return self
    def add_restriction_prolongation(self, restriction_t, prolongation_t):
        self.restriction_t = restriction_t
        self.prolongation_t = prolongation_t
        return self
    def add_tau_corrector(self, tau_t):
        self.tau_t = tau_t

    def add_neural_network(self, net):
        self.net = net
        return self

    def build_hierarchy(self):
        SGDparams = namedtuple('SGDparams', ['lr', 'momentum', 'l2_decay'])
        hierarchy_levels = []
        for level_ind in range(self.num_levels):
            optim_params = SGDparams(lr=self.learning_rate, momentum=self.momentum, l2_decay=self.weight_decay)
            loss_fn = self.loss_t()
            sgd_smoother = self.smoother_t(self.net, loss_fn = loss_fn,
                                           optim_params = optim_params, log_interval = 1)
            
            converter = self.converter_t()
            parameter_extractor = self.parameter_extractor_t(converter)
            gradient_extractor = self.gradient_extractor_t(converter)
            matching_method = self.matching_method_t()
            transfer_operator_builder = self.transfer_ops_builder_t()
            restriction = self.restriction_t(parameter_extractor, matching_method, transfer_operator_builder)
            prolongation = self.prolongation_t(parameter_extractor, restriction)

            curr_level = Level(id = level_ind,
                               presmoother = sgd_smoother,
                               postsmoother = sgd_smoother,
                               prolongation = prolongation,
                               restriction = restriction,
                               coarsegrid_solver = sgd_smoother,
                               num_epochs = self.num_epochs,
                               corrector = self.tau_t(loss_fn, gradient_extractor))
                               
            hierarchy_levels.append(curr_level)
        return hierarchy_levels

    @classmethod
    def build_from_params(cls, net, params):
        levelbuilder = cls(params["num_levels"])
        levelbuilder.add_loss(nn.MSELoss)
        levelbuilder.add_stepper_params(params["learning_rate"], params["momentum"], params["weight_decay"])
        levelbuilder.add_smoother(smoother.SGDSmoother)

        nn_is_cnn = "conv_ch" in params
        if nn_is_cnn:
            levelbuilder.add_converter(lambda : SOC.ConvolutionalConverter(net.num_conv_layers))
        else:
            levelbuilder.add_converter(SOC.MultiLinearConverter)
        
        levelbuilder.add_extractors(PE.ParamMomentumExtractor, PE.GradientExtractor)
        levelbuilder.add_matching_method(
            lambda : SimilarityMatcher.HEMCoarsener(similarity_calculator=SimilarityMatcher.StandardSimilarity(),
                                            coarsen_on_layer=None))
        levelbuilder.add_transfer_ops_builder(
            lambda : TransferOpsBuilder.PairwiseOpsBuilder_MatrixFree(weighted_projection=params["weighted_projection"]))

        levelbuilder.add_restriction_prolongation(SOR.SecondOrderRestriction, SOR.SecondOrderProlongation)

        if params["tau_corrector"] == "null":
            levelbuilder.add_tau_corrector(taucorrector.NullTau)
        elif params["tau_corrector"] == "wholeset":
            levelbuilder.add_tau_corrector(taucorrector.WholeSetTau)
        elif params["tau_corrector"] == "minibatch":
            levelbuilder.add_tau_corrector(taucorrector.MinibatchTau)

        levelbuilder.add_neural_network(net)

        neural_net_levels = levelbuilder.build_hierarchy()
        return neural_net_levels
