# !/usr/bin/env/ python
"""mg_cascadic_example.py
 * Template script to run builder and train fully-connected neural network
 * with MNIST data using prolongation operators
 * with the MTNN core
"""
# standard
import logging
import time

# PyTorch
import torch
import torch.utils as utils
from torchvision import datasets, transforms

# local source
import core
import core.configuration.reader as reader
import core.builder as builder
import core.utils.constants as constants
import core.components.smoother as smoother


logging.basicConfig(filename=(constants.EXPERIMENT_LOGS_FILENAME + ".log.txt"),
                    filemode='w',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


#####################################
# Load Data
#####################################
TRANSFORM_FN = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.ToTensor(),  # convert image to a PyTorch tensor
     transforms.Normalize((0.1307,), (0.3081,))])  # normalize with mean (tuple), standard deviation (tuple)

# Training data
TRAIN_DATASET = datasets.MNIST(root = './datasets',
                               train = True,
                               download = True,
                               transform = TRANSFORM_FN)
TRAINLOADER = utils.data.DataLoader(TRAIN_DATASET,
                                    batch_size = 10,
                                    shuffle = True,
                                    num_workers = 2)  # multi-process data loading

# Testing data
TEST_DATASET = datasets.MNIST(root = './datasets',
                              train = False,
                              download = True,
                              transform = TRANSFORM_FN)
TESTLOADER = utils.data.DataLoader(TEST_DATASET,
                                   batch_size = 10,
                                   shuffle = False,
                                   num_workers = 2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#######################################
# Instantiate Model with YAML
#######################################
# Read in YAML
yaml_file = ('path_to_yaml_file.yaml')

myyaml = reader.read(yaml_file)

mtnn_yamlmodel = builder.model.build(myyaml.model, visualize=False, debug=True)

# Build Optimizer.
optimizer = builder.optimizer.build(myyaml.optimizer)

# Build Trainer.
trainer = builder.trainer.build(myyaml.trainer, optimizer)

training_alg = core.alg.CascadicMG.CascadicMG(trainer)

print('Starting Training')

start = time.perf_counter()
mynn_yamlmodel = training_alg.train(mtnn_yamlmodel, TRAINLOADER)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))

#######################################
# Instantiate Model without YAML
#######################################
mtnn_yamlmodel = builder.model.build(myyaml.model, visualize=False, debug=True)

# Build Optimizer.
optimizer = builder.optimizer.build(myyaml.optimizer)

# Build Trainer.
trainer = builder.trainer.build(myyaml.trainer, optimizer)

training_alg = core.alg.CascadicMG.CascadicMG(trainer)

print('Starting Training')

start = time.perf_counter()
mynn_yamlmodel = training_alg.train(mtnn_yamlmodel, TRAINLOADER)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))

#####################################
# Evaluate 
#####################################
evaluator = core.evaluator.BasicEvaluator()
print("\nNet")
evaluator.evaluate_output(model=mtnn_yamlmodel, dataset=TEST_DATASET)
