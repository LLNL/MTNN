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


# Global Variables for /tests
# Directory paths to store config files.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(ROOT_DIR + "/tests/config/")

if not os.path.exists(CONFIG_DIR):
    # TODO: Sanitize file path
    os.mkdir(CONFIG_DIR)

# Edit this.
CONFIG_PARAMETERS = {
    "layers": (1, 3),  # (min, max)
    "neurons": (1, 2),
    "input": (1, 1),
}


