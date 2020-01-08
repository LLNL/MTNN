"""The MTNN prototype code"""
import sys
import os
# Check for Python 3
if sys.version_info[0] != 3:
    raise ImportError('Python 3 is required')

# Import stuff.
from MTNN.basic_evaluator import *
from MTNN.cascadic_mg_alg import *
from MTNN.identity_interpolator import *
from MTNN.sgd_training import *
from MTNN.training_alg_smoother import *
from MTNN.model import *
from MTNN.prolongation import *

################################################
# Global Variables for config_generator.py
################################################
# Directory paths to store config files.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(ROOT_DIR + "/tests/config/")

if not os.path.exists(CONFIG_DIR):
    # TODO: Sanitize file path
    os.mkdir(CONFIG_DIR)

# Edit this.
CONFIG_HYPERPARAMETERS = {
    "layers": (1, 3),  # (min, max)
    "neurons": (1, 2),
    "input": (1, 1),
}

CONFIG_LAYER_PARAMETERS = {
    'layer_type': "linear",
    'activation_type':"relu",
    'dropout':False
}

CONFIG_MODEL_PARAMETERS = {
    "model_type": "fully-connected",
    "objective_function": "mseloss",
    "optimization_method": "SGD"
}


################################################
# Global Variables for conftest.py
################################################
# Edit this.
TEST_FN_PARAMETERS = {'n_samples': 10,
                     'n_features': 1,
                     'noise': 0.1
                     }