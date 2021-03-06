# Typical execution:
# (Convolutional)
# python poisson_example.py num_levels=2 num_cycles=100 smooth_iters=4 conv_ch=1,100,200,400 conv_kernel_width=1,11,7,3 conv_stride=1,2,1,1 fc_width=3600,1024,1024 momentum=0.9 learning_rate=0.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True rand_seed=0
#
# (Fully Connected)
# python poisson_example.py num_levels=2 num_cycles=100 smooth_iters=4 fc_width=400,400 momentum=0.9 learning_rate=.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True

import torch
import torch.nn.functional as F
import numpy as np
import sys
from MTNN import models
from MTNN.components import subsetloader
from MTNN.HierarchyBuilder import HierarchyBuilder
import MTNN.MultilevelCycle as mc
from MTNN.utils.ArgReader import MTNNArgReader
from MTNN.utils.validation_callbacks import RealValidationCallback

arg_reader = MTNNArgReader()
params = arg_reader.read_args(sys.argv)

# At logging level WARNING, anything logged as log.warning() will print
# At logging level INFO, anything logged as log.warning() or log.info() will print
# At logging level DEBUG, anything logged as log.warning(), log.info(), or log.debug() will print
from MTNN.utils import logger
log = logger.create_MTNN_logger("MTNN", logging_level="INFO", log_filename=params["log_filename"])
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

import PDEDataSet
percent_train = 0.9
train_batch_size = 200
train_loader, test_loader = PDEDataSet.get_loaders(
   percent_train, train_batch_size, flatten = True, filename = "./datasets/poisson_data/Poisson4.npz")

#=====================================
# Set up network architecture
#=====================================

nx = 32 # we're using a 32 x 32 grid
nn_is_cnn = "conv_ch" in params
if nn_is_cnn:
    conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
    net = models.ConvolutionalNet(conv_info, params["fc_width"] + [nx*nx], F.relu, lambda x : x)
else:
    net = models.MultiLinearNet([3*nx*nx] + params["fc_width"] + [nx*nx], F.relu, lambda x : x)
net.log_model()

#=====================================
# Build Multigrid Hierarchy
#=====================================

neural_net_levels = HierarchyBuilder.build_standard_from_params(net, params)

#=====================================
# Run Multigrid Trainer
#=====================================

validation_callback = RealValidationCallback(test_loader, params["num_levels"], 1)
log.info("\nTesting performance prior to training...")
validation_callback(neural_net_levels, -1)
log.info("\n")
mc = mc.VCycle(neural_net_levels, cycles = params["num_cycles"],
                      subsetloader = subsetloader.NextKLoader(params["smooth_iters"]),
                      validation_callback=validation_callback)
mc.run(dataloader=train_loader)


#=====================================
# Test
#=====================================

log.info('\nTraining Complete. Testing...')
# Could use a different callback with testing instead of validation data
validation_callback(neural_net_levels, "finished")
