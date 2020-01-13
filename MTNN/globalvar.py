""" MTNN/globalvar.py
Global Variables for MTNN package
"""
import os

################################################
# Global Variables for config_generator.py
################################################

# Set config file directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CONFIG_DIR = os.path.abspath(ROOT_DIR + "/tests/config/")
POSITIVE_TEST_DIR = os.path.abspath(TEST_CONFIG_DIR + "/positive")

if not os.path.exists(POSITIVE_TEST_DIR):
    # TODO: Sanitize file path
    os.mkdir(POSITIVE_TEST_DIR)


# Edit this.
TEST_CONFIG_HYPERPARAMETERS = {
    "layers": (1, 3),  # (min, max)
    "neurons": (1, 2),
    "input": (1, 1),
}

# Edit this.
TEST_CONFIG_LAYER_PARAMETERS = {
    'layer_type': "linear",
    'activation_type':"relu",
    'dropout': False
}

# Edit this.
TEST_CONFIG_MODEL_PARAMETERS = {
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

################################################
# Global Variables for model.py
################################################
CONFIG_DIR = os.path.abspath(ROOT_DIR + "/config")
DEFAULT_CONFIG = os.path.abspath(CONFIG_DIR + "/fullyconnected.yaml")

