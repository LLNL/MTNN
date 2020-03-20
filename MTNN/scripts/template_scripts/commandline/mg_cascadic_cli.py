# !/usr/bin/env/ python
"""mtnnpython/scripts/template_scripts/mg_cascadic_cli.py
 * Template script to run builder and train fully-connected neural network
 * with MNIST data using prolongation operators
 * with the MTNN core
"""

# standard
import os

# pytorch
import torch.utils as utils
from torchvision import datasets, transforms


# local source
import MTNN
from core import builder
from scratches.codebase import trainer, constants
from configuration import reader

######################################
# Read from YAML Configuration File
#####################################
# If called from main()
try:
    CONFIG_PATH = locals()['config_path']

except KeyError:
    # Else use default MNIST_mgbase.yaml
    CONFIG_PATH = os.path.abspath(os.path.join(constants.CONFIG_DIR, 'mnist_mgbase.yaml'))

conf = reader.YamlConfig(CONFIG_PATH)
BATCH_SIZE_TRAIN = conf.get_property('batch_size_train')
BATCH_SIZE_TEST = conf.get_property('batch_size_test')

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

#######################################
# Instantiate Model
#######################################
print("Building the MTNN Model")
mtnn_model = builder.model(CONFIG_PATH)

# Build Optimizer.
optimizer = trainer.build_optimizer(CONFIG_PATH, mtnn_model.parameters())

# Set Optimizer.
mtnn_model.set_optimizer(optimizer)

mtnn_model.view_properties()


######################################
# Do training.
######################################
print("Creating the training algorithm")
smoother = MTNN.TrainingAlgSmoother(
        alg= MTNN.SGDTrainingAlg(lr=0.001, momentum=0.9, doprint=True),
        stopping= MTNN.SGDStoppingCriteria(num_epochs=1))
interp = MTNN.IdentityInterpolator()

prolongation = MTNN.LowerTriangleOperator()

training_alg = MTNN.CascadicMG(smoother=smoother, prolongation=prolongation, refinement=interp, num_levels=3)
stopping = None; # Cascadic MG is a "one-shot smoother". The input is a
                 # coarse model and the output is a fine model.

"""
print('Starting Training')
start = time.perf_counter()
net = training_alg.train(mtnn_model, TRAINLOADER, loss, stopping)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))
"""
#####################################
# Prolong
#####################################


#####################################
# Evaluate 
#####################################
