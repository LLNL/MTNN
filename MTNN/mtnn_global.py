""" MTNN/mtnn_var.py
Global variables set from commandline (run.py)
"""
# standard
import sys

# local
import MTNN.logger as mtnnlogger

# Global variables
SCRIPT_PATH = ""
CONFIG_PATH = ""
LOGFILE_NAME = ""
DEBUG = False
LOG_STDOUT = False


# Commandline paths
def set_script_path(script_path_arg: str):
    """
    Sets global variable mtnn_var.SCRIPT_PATH
    Args:
        script_path: <str> Absolute or relative path to a file or a directory.

    Returns: Null
    """
    global SCRIPT_PATH
    SCRIPT_PATH = script_path_arg


def set_config_path(config_file_arg: str):
    """
    Sets global variable  mtnn_var.CONFIG_PATH.
    Args:
        config_file: <str> Absolute or relative path to a file or a directory.

    Returns: Null
    """
    global CONFIG_PATH
    CONFIG_PATH = config_file_arg


# Commandline flags
def set_debug(debug_flag: bool):
    """
    Sets debug flag.
    Args:
        debug_flag: <bool> If true, generates logs from MTNN Model class to runs/

    Returns: Null
    """
    global DEBUG
    DEBUG = debug_flag


def set_logstdout(logstdout_flag : bool):
    """
    Sets debug flag.
    Args:
        debug_flag: <bool> If true, generates logs from MTNN Model class to runs/

    Returns: Null
    """
    global LOG_STDOUT
    LOG_STDOUT = logstdout_flag
    sys.stdout = mtnnlogger.StreamLogger()






