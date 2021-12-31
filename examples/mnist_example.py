# MNIST is a standard image classification dataset. This example shows
# how to perform multilevel training with classificaiton data.

# Typical execution
# (Fully-connected, 2 levels)
# python mnist_example.py num_levels=2 num_cycles=200 smooth_iters=4 fc_width=1024,1024,1024 momentum=0.9 learning_rate=0.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True rand_seed=0
#
# (Fully-connected, 1 level)
# python mnist_example.py num_levels=1 num_cycles=200 smooth_iters=4 fc_width=1024,1024,1024 momentum=0.9 learning_rate=0.01 weight_decay=1e-6 tau_corrector=wholeset weighted_projection=True rand_seed=0

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as td
import sys
from os import path
from MTNN import models
from MTNN.components import subsetloader
from MTNN.HierarchyBuilder import HierarchyBuilder
import MTNN.MultilevelCycle as mc
from MTNN.utils.ArgReader import MTNNArgReader
from MTNN.utils.validation_callbacks import ClassifierValidationCallback

arg_reader = MTNNArgReader()
params = arg_reader.read_args(sys.argv)

class MnistData:
    """
    Loads Pytorch Mnist Dataset into Dataloaders
    Image size is 28 x 28
        - 60,000 training images
        -10,000 testing images
    """
    preprocess = transforms.Compose(
        [transforms.ToTensor(), # Convert to 3 Channels
         transforms.Normalize((0.1307,), (0.3081,))])  # mean, standard deviation

    def __init__(self, trainbatch_size, testbatch_size, root="./datasets"):
        self.trainset = datasets.MNIST(root=root,
                                       train=True,
                                       download=True,
                                       transform=MnistData.preprocess)
        self.testset = datasets.MNIST(root=root,
                                      train=False,
                                      download=True,
                                      transform=MnistData.preprocess)
        self.trainloader = td.DataLoader(self.trainset,
                                        batch_size=trainbatch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=True)
        self.testloader = td.DataLoader(self.testset,
                                        batch_size=testbatch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        pin_memory=True)

# Load Data and Model
dataset = MnistData(trainbatch_size=200, testbatch_size=1000)
train_loader = dataset.trainloader
test_loader = dataset.testloader

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

#============================
# Set up network architecture
#============================

nn_is_cnn = "conv_ch" in params
#output_activation = lambda x : F.log_softmax(x, dim=0)
if nn_is_cnn:
    conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
    net = models.ConvolutionalNet(conv_info, params["fc_width"] + [10], F.relu, F.log_softmax)
else:
    net = models.MultiLinearNet([784] + params["fc_width"] + [10], F.relu, F.log_softmax)
net.log_model()

#=====================================
# Build Multigrid Hierarchy
#=====================================

neural_net_levels = HierarchyBuilder.build_standard_from_params(net, params, loss=nn.CrossEntropyLoss)

#=====================================
# Run Multigrid Trainer
#=====================================

validation_callback = ClassifierValidationCallback(test_loader, params["num_levels"], val_frequency=2)

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

#=====================================
# Test
#=====================================

log.warning('\nTraining Complete. Testing...')
# Could use a different callback with testing instead of validation data
validation_callback(neural_net_levels, "finished")
