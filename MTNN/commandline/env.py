""" MTNN/env.py
Global variables set from commandline (run.py)
"""
# standard
import os
import datetime

from . import utils

# Commandline arguments
SCRIPT_PATH = ""
CONFIG_PATH = ""
DIR_PATH = ""
LOGFILE_NAME = ""
DEBUG = False
LOG_STDOUT = False

EXPERIMENT_LOGS_DIR = os.path.abspath(os.path.join(utils.get_caller_filepath() + "runs/logs/"))

if not os.path.exists(EXPERIMENT_LOGS_DIR):
    os.makedirs(EXPERIMENT_LOGS_DIR)

EXPERIMENT_LOGS_FILENAME = os.path.join(EXPERIMENT_LOGS_DIR + "/" + utils.get_caller_filename() + "_"
                                        + datetime.datetime.today().strftime("%m%d%Y") + "_"
                                        + datetime.datetime.now().strftime("%H:%M:%S") + "_"
                                        + datetime.datetime.today().strftime("%A"))


