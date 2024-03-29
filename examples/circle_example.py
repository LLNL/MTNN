# This is a synthetic example that visually shows the benefits of
# multilevel training. The plotting function shows the true function,
# neural network output values, and residual. Colored arrows show the
# activation position of the ReLU function on the neurons in the first
# layer. Blue arrows indicate that the neuron has a positive effect on
# function value in that direction, while a red arrow indicates a
# negative effect.

# It is worth trying different hierarchy depths, as well as both
# tau_corrector=null and tau_corrector=wholeset, which imply greater
# or lesser multilevel-induced regularization.

# Example execution:
# python circle_example.py rand_seed=1 fc_width=24 learning_rate=0.1 momentum=0.7 weight_decay=1e-9 weighted_projection=False num_cycles=40000 num_levels=3 tau_corrector=none smooth_iters=200
#
# Some other options include
# python circle_example.py rand_seed=1 fc_width=24 learning_rate=0.1 momentum=0.9 weight_decay=1e-9 weighted_projection=True num_cycles=4000 num_levels=3 tau_corrector=none smooth_iters=200
#
# python circle_example.py rand_seed=0 fc_width=24 learning_rate=0.1 momentum=0.7 weight_decay=1e-9 weighted_projection=False num_cycles=4000 num_levels=3 tau_corrector=wholeset smooth_iters=10
#
# python circle_example.py rand_seed=0 fc_width=24 learning_rate=0.1 momentum=0.9 weight_decay=1e-9 weighted_projection=True num_cycles=8000 num_levels=1 tau_corrector=none

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys
from MTNN.architectures.MultilinearModel import MultilinearNet
from MTNN.components import subsetloader
from MTNN.HierarchyBuilder import HierarchyBuilder
import MTNN.MultilevelCycle as mc
from MTNN.utils.ArgReader import MTNNArgReader
from MTNN.utils.validation_callbacks import RealValidationCallback
from MTNN.utils import deviceloader
from utils_for_circle_example import CircleHelper


arg_reader = MTNNArgReader()
params = arg_reader.read_args(sys.argv)

from MTNN.utils import logger
# At logging level WARNING, anything logged as log.warning() will print
# At logging level INFO, anything logged as log.warning() or log.info() will print
# At logging level DEBUG, anything logged as log.warning(), log.info(), or log.debug() will print
log = logger.create_MTNN_logger("MTNN", logging_level="WARNING", log_filename=params["log_filename"])
log.warning("Input parameters:\n{}\n".format(params))

# For reproducibility. Comment out for possibly-improved efficiency
# but without reproducibility.
torch.manual_seed(params["rand_seed"])
np.random.seed(params["rand_seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#=======================================
# Set up network architecture
#=======================================

net = MultilinearNet([2] + params["fc_width"] + [1], F.relu, lambda x : x)

#=====================================
# Build Multigrid Hierarchy
#=====================================

neural_net_levels = HierarchyBuilder.build_standard_from_params(net, params)
        
#=====================================
# Get data, plot initial performance
#=====================================

log.info("\nTesting performance prior to training...")
circle_helper = CircleHelper(num_train_samples=400, num_test_samples=900)
train_loader, test_loader = circle_helper.get_dataloaders()
print("Plotting target function, initial neural network function, and initial residual.")
print("Red arrows represent ReLU activation points with negative effect on the function value.")
print("Blue arrows represent ReLU activation points with positive effect on the function value.")
circle_helper.plot_outputs(neural_net_levels[0].net, 0)

validation_callbacks = [RealValidationCallback("Circle Validation", test_loader, params["num_levels"], 1)]
validation_callbacks[0](neural_net_levels, -1)
log.info("\n")

#=====================================
# Training
#=====================================

mc = mc.VCycle(neural_net_levels, cycles = params["num_cycles"],
                      subsetloader = subsetloader.WholeSetLoader(), # Note this subsetloader is different from Darcy and Poisson examples
                      validation_callbacks=validation_callbacks)
mc.run(dataloader=train_loader)


#===============================
# Test
#===============================

log.info('\nTraining Complete. Testing...')
# Could use a different callback with testing instead of validation data
validation_callbacks[0](neural_net_levels, "finished")

coarse_net = neural_net_levels[1].net if params["num_levels"] > 1 else None
for i, level in enumerate(neural_net_levels):
    print("Plotting neural network function and residual at hierarchy level {}".format(i))
    circle_helper.plot_outputs(level.net, i)
