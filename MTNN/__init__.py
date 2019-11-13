"""The MTNN prototype code"""
import sys

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
from MTNN.randomperturb import *
