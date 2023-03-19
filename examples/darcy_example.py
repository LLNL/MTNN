# The Darcy flow equations model fluid flow through a porous medium,
# such as subsurface water flow. In this example, we train a neural
# network to take as input the permeability field of the 2D, square
# space and to estimate as output the fluid flux through the
# right-hand boundary of the space.


# Typical exeuction:
# (Convolutional)
# python darcy_example.py num_levels=2 num_cycles=100 smooth_iters=4 conv_ch=1,100,200,400 conv_kernel_width=1,11,7,3 conv_stride=1,2,1,1 fc_width=3600,1024,1024 momentum=0.9 learning_rate=0.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True rand_seed=0
#
# (Fully-connected)
# python darcy_example.py num_levels=2 num_cycles=100 smooth_iters=4 fc_width=1024,1024,1024 momentum=0.9 learning_rate=0.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True rand_seed=0

import torch
import numpy as np
import torch.nn.functional as F
import sys
from os import path
from MTNN import models
from MTNN.components import subsetloader
from MTNN.HierarchyBuilder import HierarchyBuilder
import MTNN.MultilevelCycle as mc
from MTNN.utils.ArgReader import MTNNArgReader
from MTNN.utils.validation_callbacks import RealValidationCallback, SaveParamsCallback

arg_reader = MTNNArgReader()
params = arg_reader.read_args(sys.argv)

# At logging level WARNING, anything logged as log.warning() will print
# At logging level INFO, anything logged as log.warning() or log.info() will print
# At logging level DEBUG, anything logged as log.warning(), log.info(), or log.debug() will print
from MTNN.utils import logger
log = logger.create_MTNN_logger("MTNN", logging_level="WARNING", log_filename=params["log_filename"])
log.warning("Input parameters:\n{}\n".format(params))

# For reproducibility. Comment out for possibly-improved efficiency
# but without reproducibility.
torch.manual_seed(params["rand_seed"])
np.random.seed(params["rand_seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#=======================================
# Set up data
#=====================================

import DarcyDataset
percent_train = 0.8
train_batch_size = 200
train_loader, test_loader = DarcyDataset.get_loaders(percent_train, train_batch_size)

#=====================================
# Set up network architecture
#=====================================

nn_is_cnn = "conv_ch" in params
if nn_is_cnn:
    conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
    net = models.ConvolutionalNet(conv_info, params["fc_width"] + [1], F.relu, lambda x : x)
else:
    net = models.MultiLinearNet([1024] + params["fc_width"] + [1], F.relu, lambda x : x)

if "load_params_from" in params:
    net.load_params(params["load_params_from"])

net.log_model()

#=====================================
# Build Multigrid Hierarchy
#=====================================

neural_net_levels = HierarchyBuilder.build_standard_from_params(net, params)

#=====================================
# Run Multigrid Trainer
#=====================================

train_loader2, test_loader2 = DarcyDataset.get_loaders(percent_train, train_batch_size)
callbacks = [RealValidationCallback("Validation_Data", test_loader2, params["num_levels"], 10),
             RealValidationCallback("Training_Data", train_loader2, params["num_levels"], 10)]
if "save_params_at" in params:
    callbacks.append(SaveParamsCallback(params["save_params_at"]))

log.info("\nTesting performance prior to training...")
for c in callbacks:
    c(neural_net_levels, -1)

log.info("\n")
mc = mc.VCycle(neural_net_levels, cycles = params["num_cycles"],
               subsetloader = subsetloader.NextKLoader(params["smooth_iters"]),
               validation_callbacks=callbacks)
mc.run(dataloader=train_loader)


#=====================================
# Test
#=====================================

log.info('\nTraining Complete. Testing...')
# Could use a different callback with testing instead of validation data
for c in callbacks:
    c(neural_net_levels, "finished")
