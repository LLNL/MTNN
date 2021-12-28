import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sys
from os import path
sys.path.append("../")

from MTNN.utils import logger
# At logging level WARNING, anything logged as log.warning() will print
# At logging level INFO, anything logged as log.warning() or log.info() will print
# At logging level DEBUG, anything logged as log.warning(), log.info(), or log.debug() will print
log = logger.create_MTNN_logger("MTNN", logging_level="INFO", write_to_file=False)

from MTNN.core.components import models, subsetloader
from MTNN.core.multigrid.level import HierarchyBuilder
import MTNN.core.multigrid.scheme as mg
from MTNN.utils.ArgReader import ArgReader
from MTNN.utils.validation_callbacks import RealValidationCallback

# Typical exeuction:
# python darcy_test.py num_levels=2 num_cycles=100 smooth_iters=4 conv_ch=1,100,200,400 conv_kernel_width=1,11,7,3 conv_stride=1,2,1,1 fc_width=3600,1024,1024 momentum=0.9 learning_rate=0.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True rand_seed=0

arg_reader = ArgReader()
params = arg_reader.read_args(sys.argv)
print("Input parameters:")
print("{}\n".format(params))

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
net.log_model()

#=====================================
# Build Multigrid Hierarchy
#=====================================

neural_net_levels = HierarchyBuilder.build_standard_from_params(net, params)

#=====================================
# Run Multigrid Trainer
#=====================================

validation_callback = RealValidationCallback(test_loader, params["num_levels"], 1)
mg_scheme = mg.VCycle(neural_net_levels, cycles = params["num_cycles"],
                      subsetloader = subsetloader.NextKLoader(params["smooth_iters"]),
                      validation_callback=validation_callback)
mg_scheme.run(dataloader=train_loader)


#=====================================
# Test
#=====================================

print('\nTraining Complete. Testing...')
# Could use a different callback with testing instead of validation data
validation_callback(neural_net_levels)
