"""
MTNN/mtnn_defaults.py
Global default variables for MTNN package
"""
import os
import sys
import re
import datetime
import inspect


# TODO: Clean
################################################
# Global Variables for config_generator.py
################################################
# Set config file directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CONFIG_DIR = os.path.abspath(ROOT_DIR + "/tests/config/")
POSITIVE_TEST_DIR = os.path.abspath(TEST_CONFIG_DIR + "/positive")

if not os.path.exists(POSITIVE_TEST_DIR):
    # TODO: Sanitize file path
    os.makedirs(POSITIVE_TEST_DIR)

TEST_CONFIG_MODEL_TYPE = {
    "model_type": "fully-connected",
    "input_size": 2
}

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
TEST_CONFIG_MODEL_HYPERPARAMETERS = {
    "log_interval": 10,
    "num_epochs": 10,
    "batch_size_train": 10,
    "batch_size_test": 10,
    "learning_rate": 0.01,
    "momentum": 0.5,
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
DEFAULT_CONFIG = os.path.abspath(CONFIG_DIR + "/fullyconnected_template.yaml")

################################################
# Helper functions and Global Variables for hello_model_sklearn.py
################################################
# TODO: Refactor: move to pathfinder?
def get_caller_filename():
    """
    Gets previous caller's filepath.
    Returns:
        prev_caller_filepath <str>: Previous caller's filename

    """
    # Get the previous caller's stack frame and extract its file path
    last_frame_info = inspect.stack()[-1]
    caller_filepath = last_frame_info[1]  # in python 3.5+, you can use frame_info.filename
    del last_frame_info  # drop the reference to the stack frame to avoid reference cycles

    prev_caller_filename = os.path.basename(caller_filepath).strip(".py")
    return prev_caller_filename

def get_caller_filepath():
    """
    Gets previous caller's filepath.
    Returns:
        clean_caller_filepath <str>: Previous caller's absolute file path

    """
    # Get the previous caller's stack frame and extract its file path.
    last_frame_info = inspect.stack()[-1]
    caller_filepath = last_frame_info[1]  # in python 3.5+, you can use frame_info.filename
    del last_frame_info  # drop the reference to the stack frame to avoid reference cycles

    # Use regex to get the base filepath.
    filename_match = re.search( "\w*.py$", caller_filepath)
    clean_caller_filepath = (caller_filepath).strip(filename_match.group())
    return clean_caller_filepath


# TODO: Refactor: Redirect to the /runs to the directory of the script caller.
#  Currently if calling a script from main, creates /run in MTNN/
EXPERIMENT_LOGS_DIR = os.path.abspath(os.path.join(get_caller_filepath() + "runs/logs/"))

if not os.path.exists(EXPERIMENT_LOGS_DIR):
    os.makedirs(EXPERIMENT_LOGS_DIR)

EXPERIMENT_LOGS_FILENAME = os.path.join(EXPERIMENT_LOGS_DIR + "/" + get_caller_filename() + "_"
                                        + datetime.datetime.today().strftime("%m%d%Y") + "_"
                                        + datetime.datetime.now().strftime("%H:%M:%S") + "_"
                                        + datetime.datetime.today().strftime("%A"))




