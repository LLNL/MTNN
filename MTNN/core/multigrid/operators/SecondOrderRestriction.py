"""
Restriction Operators
"""
import torch
import torch.nn as nn
import numpy as np
import collections as col
from abc import ABC, abstractmethod

# local
import MTNN.core.multigrid.scheme as mg
#import MTNN.core.multigrid.operators.interpolator as interp
import MTNN.core.components.models as models
from MTNN.utils.datatypes import operators
import MTNN.utils.logger as log
import MTNN.utils.printer as printer
import MTNN.utils.deviceloader as deviceloader

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['SecondOrderRestriction',
           'SecondOrderProlongation']

############################################
# Data types
#############################################

"""ParamLibrary - a data store of parameters associated with a neural network

weights_list and bias_list are each lists of length equal to the
number of layers in a network.

Each element of weights_list is a tensor of order >=2 such that
dimension -2 is the number of output channels and dimension -1 is the
number of input channels. Note that for fully-connected layers, this
tensor will be of order 2 exactly, in which case dimension -2 is the
rows and dimension -1 is the columns.

Each element of bias_list is a tensor of order 1 of length equal to
the number of output channels.
"""
ParamLibrary = col.namedtuple("ParamLibrary", "weights biases")


"""CoarseMapping - specifies a mapping from fine channels to coarse channels.

fine2coarse_map <List(List)> - A list of length equal to the number of
network layers. Each element is itself a list, of length equal to the
number of channels in the fine-level network. Each element specifies
the index of the coarse channel to which this fine channel is mapped.

num_coarse_channels <List> - A list of length equal to the number of
network layers. Each element contains the number of coarse channels at
that layer.
"""
CoarseMapping = col.namedtuple("CoarseMapping", "fine2coarse_map, num_coarse_channels")


"""TransferOps - The matrix operators used in restriction and prolongation.

R_ops <List> - A list, of length equal to the number of layers minus
1, containing the R matrices used in restriction.

P_ops <List> - A list, of length equal to the number of layers minus
1, containing the P matrices used in restriction.

R_for_grad_ops <List> - A list, of length equal to the number of
layers minus 1, containing the R matrices used in restriction of
gradients in a tau correction.

P_for_grad_ops <List> - A list, of length equal to the number of
layers minus 1, containing the P matrices used in restriction of
gradients in a tau correction.
"""
TransferOps = col.namedtuple("TransferOps", "R_ops P_ops R_for_grad_ops P_for_grad_ops")


####################################################################
# Transfer Operator Templates
####################################################################

def transfer(Wmats, Bmats, R_ops, P_ops):
    num_layers = len(Wmats)
    Wdest_array = []
    Bdest_array = []
    for layer_id in range(num_layers):
        Wsrc = Wmats[layer_id]
        Bsrc = Bmats[layer_id]
        if layer_id < num_layers - 1:
            if layer_id == 0:
                Wdest = R_ops[layer_id] @ Wsrc
#                print(layer_id, "Before: ", R_ops[layer_id].shape, Wsrc.shape, "After: ", Wdest.shape)
            else:
#                print(layer_id, "Before: ", R_ops[layer_id].shape, Wsrc.shape, P_ops[layer_id-1].shape, end=" ")
                Wdest = R_ops[layer_id] @ Wsrc @ P_ops[layer_id - 1]
#                print("After: ", Wdest.shape)
            Bdest = R_ops[layer_id] @ Bsrc
        elif layer_id > 0:            
            Wdest = Wsrc @ P_ops[layer_id-1]
#            print(layer_id, "Before: ", Wsrc.shape, P_ops[layer_id-1].shape, "After: ", Wdest.shape)
            Bdest = Bsrc.clone()

        Wdest_array.append(Wdest)
        Bdest_array.append(Bdest)
    return Wdest_array, Bdest_array

def transfer_star(Wmats, Bmats, R_ops, P_ops):
    num_layers = len(Wmats)
    Wdest_array = []
    Bdest_array = []
    for layer_id in range(num_layers):
        Wsrc = Wmats[layer_id]
        Bsrc = Bmats[layer_id]
        if layer_id < num_layers - 1:
            if layer_id == 0:
                Wdest = R_ops[layer_id] * Wsrc
            else:
                Wdest = R_ops[layer_id] * Wsrc * P_ops[layer_id - 1]
            Bdest = R_ops[layer_id] * Bsrc
        elif layer_id > 0:
            Wdest = Wsrc * P_ops[layer_id-1]
            Bdest = Bsrc.clone()

        Wdest_array.append(Wdest)
        Bdest_array.append(Bdest)
    return Wdest_array, Bdest_array

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

    def __init__(self, parameter_extractor, matching_method, transfer_operator_builder):
        """Construct the SecondOrderRestrcition.

        Inputs: 
        parameter_extractor <ParameterExtractor>

        matching_method <Callable>. Takes as input a ParamLibrary and
        a Level, produces as output a CoarseMapping object.

        transfer_operator_builder <Callable>. Takes as input a
        ParamLibrary, a CoarseMapping, and a torch.device, and
        produces a TransferOps object.
        """
        self.parameter_extractor = parameter_extractor
        self.matching_method = matching_method
        self.transfer_operator_builder = transfer_operator_builder

    def apply(self, fine_level, coarse_level, dataloader, verbose=False):
        fine_param_library, fine_momentum_library = self.parameter_extractor.extract_from_network(fine_level)

        coarse_mapping = self.matching_method(fine_param_library, fine_level.net)

        self.transfer_ops = self.transfer_operator_builder(fine_param_library,
                                                           coarse_mapping,
                                                           deviceloader.get_device())
        R_ops = self.transfer_ops.R_ops
        P_ops = self.transfer_ops.P_ops

        W_f_array, B_f_array = fine_param_library
        W_c_array, B_c_array = transfer(W_f_array, B_f_array, R_ops, P_ops)

        mW_f_array, mB_f_array = fine_momentum_library
        mW_c_array, mB_c_array = transfer(mW_f_array, mB_f_array, R_ops, P_ops)

        coarse_param_library = ParamLibrary(W_c_array, B_c_array)
        coarse_momentum_library = ParamLibrary(mW_c_array, mB_c_array)

        # So this is some terrible software design right here. :-D
        # TODO: Refactor for architecture extensibility.
        if fine_level.net.__class__.__name__ == "MultiLinearNet":
            coarse_level_dims = [fine_level.net.layers[0].in_features] + coarse_mapping.num_coarse_channels + \
                [fine_level.net.layers[-1].out_features]
            coarse_level.net = fine_level.net.__class__(coarse_level_dims,
                                                        fine_level.net.activation,
                                                        fine_level.net.output)
            coarse_level.net.set_device(fine_level.net.device)
        elif fine_level.net.__class__.__name__ == "ConvolutionalNet":
            num_conv_layers = fine_level.net.num_conv_layers
            out_ch, kernel_widths, strides = zip(*fine_level.net.conv_channels)
            out_ch = [out_ch[0]] + coarse_mapping.num_coarse_channels[:num_conv_layers]
            conv_channels = list(zip(out_ch, kernel_widths, strides))
            fc_dims = [conv_channels[-1][0] * W_c_array[num_conv_layers].shape[0]] + \
                      coarse_mapping.num_coarse_channels[num_conv_layers:] + \
                      [fine_level.net.layers[-1].out_features]
            coarse_level.net = fine_level.net.__class__(conv_channels,
                                                        fc_dims,
                                                        fine_level.net.activation,
                                                        fine_level.net.output_activation)
        else:
            raise RuntimeError("SecondOrderRestriction::apply: {} is not a supported network type.".format(fine_level.net.__class__.__name__))
        
        self.parameter_extractor.insert_into_network(coarse_level, coarse_param_library,
                                                     coarse_momentum_library)

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

    def apply(self, fine_level, coarse_level, dataloader, verbose):
        assert(fine_level.id < coarse_level.id)

        R_ops = self.restriction.transfer_ops.R_ops
        P_ops = self.restriction.transfer_ops.P_ops

        coarse_param_library, coarse_momentum_library = self.parameter_extractor.extract_from_network(coarse_level)
        W_c_array, B_c_array = coarse_param_library
        mW_c_array, mB_c_array = coarse_momentum_library
        c_init_params = coarse_level.init_params
        c_init_momentum = coarse_level.init_momentum
        eW_array = []
        eB_array = []
        emW_array = []
        emB_array = []
        for layer_id in range(len(W_c_array)):
#            print("{}: W_c and Winit shapes are {}, {}"
#                  .format(layer_id, W_c_array[layer_id].shape, c_init_params.weights[layer_id].shape))
            eW_array.append(W_c_array[layer_id] - c_init_params.weights[layer_id])
            eB_array.append(B_c_array[layer_id] - c_init_params.biases[layer_id])
            emW_array.append(mW_c_array[layer_id] - c_init_momentum.weights[layer_id])
            emB_array.append(mB_c_array[layer_id] - c_init_momentum.biases[layer_id])
        eW_array, eB_array = transfer(eW_array, eB_array, P_ops, R_ops)
        emW_array, emB_array = transfer(emW_array, emB_array, P_ops, R_ops)

        fine_param_diff_library = ParamLibrary(eW_array, eB_array)
        fine_momentum_diff_library = ParamLibrary(emW_array, emB_array)
        self.parameter_extractor.add_to_network(fine_level, fine_param_diff_library,
                                                fine_momentum_diff_library)
