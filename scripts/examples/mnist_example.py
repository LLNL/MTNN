# !/usr/bin/env/ python
"""
Test script to run build and train fully-connected neural network
with MNIST data using prolongation operators
"""

# standard library packages
import os
import sys
import datetime
import argparse
import time
import yaml
import inspect  # For testing.

# Pytorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torchvision import datasets, transforms

# local source
import MTNN
from MTNN import model as mf

# Set training hyper-parameters
N_EPOCHS = 2  # Set for testing
BATCH_SIZE_TRAIN = 100  # Set for testing
BATCH_SIZE_TEST = 500  # Set for testing
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10  # Set for testing
OBJECTIVE_FN = nn.CrossEntropyLoss()

# CUDA
RANDOM_SEED = 1
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.enabled = False  # Disable for repeatability

# TODO: Get mean and std deviation on data sets to pass to TRANSFORM_FN

# Load and transform data
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
                                    batch_size = BATCH_SIZE_TRAIN,
                                    shuffle = True,
                                    num_workers = 2)  # multi-process data loading

# Testing data
TEST_DATASET = datasets.MNIST(root = './datasets',
                              train = False,
                              download = True,
                              transform = TRANSFORM_FN)
TESTLOADER = utils.data.DataLoader(TEST_DATASET,
                                   batch_size = BATCH_SIZE_TEST,
                                   shuffle = False,
                                   num_workers = 2)


def parse_args():
    """
    Returns: commandline_args
    """
    parser = argparse.ArgumentParser(description = "Runs MTNN with a YAML config file")
    parser.add_argument("-f", "--file",
                        required = True,
                        nargs = 1,
                        type = str,
                        action = "store",
                        dest = "filepath",
                        default = sys.stdin,
                        help = "Specify path to a YAML configuration file")
    commandline_args = parser.parse_args()
    return commandline_args


def clean(filepath):
    """
    Sanitize filepath input
    Args:
        filepath (str)

    Returns:
        clean_filepath(str)
    """
    clean_filepath = filepath.strip()
    return clean_filepath


def check_config(filepath: str) -> bool:
    """
    Check if config file can be read without errors, is not empty and is well-formed
    Args:
        filepath (str) Full path to the YAML configuration file
    Returns: bool
    """
    try:
        if os.path.isfile(filepath):
            # TODO: Validate YAML config file. Use Python confuse API?
            return True
    except TypeError as exc:
        print(exc)
        sys.exit(1)



# Parse arguments
args = parse_args()
file_path = clean(args.filepath[0])
config_file = file_path

# Training and test loss counters
test_losses = []
test_counter = [i * len(TRAINLOADER.dataset) for i in range(N_EPOCHS + 1)]

# Check if config file is NN kosher.
try:
    check_config(config_file)
except ValueError:
    print("Configuration file is not well-formed.")
    sys.exit(1)

# Check if config file is YAML kosher.
with open(config_file, "r") as file_stream:
    try:
        ###########################################################################
        # Instantiate the neural network (NN) model.
        ###########################################################################
        print("Setting up the network.")

        model_config = yaml.load(file_stream, Loader = yaml.SafeLoader)
        model = mf.Model(tensorboard = True, debug = True)
        model.set_config(model_config)
        print(model)
        optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

    except yaml.YAMLError as error:
        print(error)

    ################################################################################
    # Modify training algorithm
    #################################################################################
    # From examples/mg_example.py

    SMOOTHER = MTNN.TrainingAlgSmoother(
        alg = MTNN.SGDTrainingAlg(lr = 0.001, momentum = 0.9, doprint = True),
        stopping = MTNN.SGDStoppingCriteria(num_epochs = 1))

    # Interpolator should take old model and create new model with new size
    # Example: “interp.Apply(oldModel, newModel)
    # TODO: Interpolator.py
    # TODO: Prolongation.py
    INTERP = MTNN.IdentityInterpolator()


    # TODO: FIX
    PROLONGATION_OPERATOR = MTNN.RandomPerturbation(prolongation = "randomperturb")

    TRAINING_ALG = MTNN.CascadicMG(smoother = SMOOTHER,
                                   refinement = INTERP,
                                   prolongation = PROLONGATION_OPERATOR,
                                   num_levels = 3)
    STOPPING = None  # Cascadic MG is a "one-shot solver". The input is a
    # coarse model and the output is a fine model.

    #################################################################################
    # Train
    #################################################################################
    print('Starting Training')

    START = time.perf_counter()
    # model = TRAINING_ALG.train(model, TRAINLOADER, OBJECTIVE_FN, STOPPING) # Cascadic_alg

    model.set_training_parameters(obj_fn = OBJECTIVE_FN,
                                    opt = optimizer)
    model.fit(dataloader = TRAINLOADER,
              num_epochs = N_EPOCHS,
              log_interval = LOG_INTERVAL,
              checkpoint = False)

    STOP = time.perf_counter()
    print('Finished Training (%.3fs)' % (STOP - START))

    #################################################################################
    # Test the NN model and evaluate algorithm
    #################################################################################

    # Test the output of trained model
    tester = MTNN.BasicEvaluator()

    print('Starting Testing')
    start = time.perf_counter()
    correct, total = tester.evaluate(model, TESTLOADER)
    stop = time.perf_counter()
    print('Finished Testing (%.3fs)' % (stop - start))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))