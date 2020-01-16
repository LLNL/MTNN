""" MTNN/tests_var.py
Global Variables for MTNN package
"""
import os
import datetime
import inspect

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
    'activation_type': "relu",
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


################################################
# Global Variables for hello_model_sklearn.py
################################################

def get_caller_filepath():
    # get the caller's stack frame and extract its file path
    frame_info = inspect.stack()[1]
    filepath = frame_info[1]  # in python 3.5+, you can use frame_info.filename
    del frame_info  # drop the reference to the stack frame to avoid reference cycles

    caller_filepath = os.path.basename(filepath)

    return caller_filepath


EXAMPLES_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir, "examples"))
EXPERIMENT_LOGS_DIR = os.path.abspath(os.path.join(EXAMPLES_DIR + "/runs/logs/"))

if not os.path.exists(EXPERIMENT_LOGS_DIR):
    # TODO: Sanitize file path
    os.mkdir(EXPERIMENT_LOGS_DIR)

EXPERIMENT_LOGS_FILENAME = os.path.join(EXPERIMENT_LOGS_DIR + get_caller_filepath() +"/"
                                    + datetime.datetime.now().strftime("%H:%M:%S"))
