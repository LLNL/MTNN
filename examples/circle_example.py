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
# python circle_test.py rand_seed=0 fc_width=24 learning_rate=0.1 momentum=0.9 weight_decay=1e-9 weighted_projection=True num_cycles=4000 num_levels=3 tau_corrector=none
#
# python circle_test.py rand_seed=0 fc_width=24 learning_rate=0.1 momentum=0.9 weight_decay=1e-9 weighted_projection=True num_cycles=4000 num_levels=3 tau_corrector=wholeset
#
# python circle_test.py rand_seed=0 fc_width=24 learning_rate=0.1 momentum=0.9 weight_decay=1e-9 weighted_projection=True num_cycles=8000 num_levels=1 tau_corrector=none

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import sys
sys.path.append("../")

from MTNN.utils import logger
# At logging level WARNING, anything logged as log.warning() will print
# At logging level INFO, anything logged as log.warning() or log.info() will print
# At logging level DEBUG, anything logged as log.warning(), log.info(), or log.debug() will print
log = logger.create_MTNN_logger("MTNN", logging_level="WARNING", write_to_file=False)

from MTNN import models
from MTNN.components import subsetloader
from MTNN.HierarchyBuilder import HierarchyBuilder
import MTNN.MultilevelCycle as mc
from MTNN.utils.ArgReader import ArgReader
from MTNN.utils.validation_callbacks import RealValidationCallback

from utils_for_circle_example import CircleHelper



arg_reader = ArgReader()
params = arg_reader.read_args(sys.argv)
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

net = models.MultiLinearNet([2] + params["fc_width"] + [1], F.relu, lambda x : x)

# Set bias to have reasonable inflection points
w = net.layers[0].weight.data
b = net.layers[0].bias.data
zp = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=len(b)).astype(np.float32))
b[:] = torch.norm(w, dim=1) * zp

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
circle_helper.plot_outputs(neural_net_levels[0].net, 1)

validation_callback = RealValidationCallback(test_loader, params["num_levels"], 1)
validation_callback(neural_net_levels)
log.info("\n")

#=====================================
# Training
#=====================================

mc = mc.VCycle(neural_net_levels, cycles = params["num_cycles"],
                      subsetloader = subsetloader.WholeSetLoader(), # Note this subsetloader is different from Darcy and Poisson examples
                      validation_callback=validation_callback)
mc.run(dataloader=train_loader)


#===============================
# Test
#===============================

log.info('\nTraining Complete. Testing...')
# Could use a different callback with testing instead of validation data
validation_callback(neural_net_levels)

coarse_net = neural_net_levels[1].net if params["num_levels"] > 1 else None
for i, level in enumerate(neural_net_levels):
    circle_helper.plot_outputs(level.net, i+1)
