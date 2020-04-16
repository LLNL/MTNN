"""
MTNN/commandline/utils.py
Helper methods for run.py
"""
# standard
import sys
import os
import pathlib
import argparse

from pathlib import PurePath

# local
from . import env
import logging.logger as logger


def parse_args() -> list:
    """
    Returns:
        commandline_args <list>
    """
    parser = argparse.ArgumentParser(description = "Runs MTNN script with a YAML configuration file")
    # Option for experiment script
    parser.add_argument("script",
                        nargs = 1,
                        type = str,
                        action = "store",
                        default = sys.stdin,
                        help = "Specify path to the test script")
    # Option for configuration file
    parser.add_argument("configuration",
                        nargs = 1,
                        type = str,
                        action = "store",
                        default = sys.stdin,
                        help = "Specify path to a YAML configuration file")
    # Flag for debugging logs to runs/
    parser.add_argument("-d", "--debug",
                        default = sys.stdin,
                        action = "store_true",
                        help = "Sets flag for debug logs")

    # Flag for logging stdout runs/
    parser.add_argument("-l", "--log",
                        default = sys.stdin,
                        action = "store_true",
                        help = "Sets flag for logging stdout")

    # TODO: Option for directory of yaml files
    # TODO: Option for checkpointing
    # TODO: Option for tensorboard visualization

    # Parse commandline arguments
    commandline_args = parser.parse_args()

    return commandline_args




def find_config(config_dir_path: str, filename: str) -> str:
    """
    Searches the base directory given by config_dir for the filename.
    Args:
        config_dir: <str> Directory path
        filename: <str> Name of the configuration file

    Returns:
        path: <str> Absolute UNIX path

    """
    dir_path = PurePath(config_dir_path)
    cwd = dir_path.parent
    print(cwd)
    results = []
    for root, dirs, files in os.walk(cwd):
        if filename in files:
            path = os.path.abspath(os.path.join(root, filename))
            results.append(path)
            return path
    if not results:
        print(f"No such file  {filename} in current directory: {cwd}")
        raise FileNotFoundError


def check_path(filepath: str) -> str:
    """
    Check if path exists and return absolute file path
    Args:
        filepath <str>
    Returns:
        abs_filepath <str>
    """
    filepath = filepath.strip()
    file = pathlib.Path(filepath)

    # Check if path exists
    try:
        if file.is_file() and file.exists():
            pass
        file = open(filepath)
        file.close()
    except IOError or FileNotFoundError as exc:
        print(exc)

    # Return absolute path
    if not os.path.isabs(filepath):
        clean_filepath = filepath.strip("./")
        abs_filepath = os.path.join(os.path.abspath('.'), clean_filepath)

    else:
        abs_filepath = filepath
    return abs_filepath



def check_config(filepath: str) -> bool:
    # TODO: Fill in. Validate YAML configuration file. Use Python confuse API?
    """
    Check if configuration file can be read without errors, is not empty and is well-formed YAML file
    Args:
        filepath (str) Full path to the YAML configuration file
    Returns: bool
    """
    try:
        pass
    except ValueError:
        print("Configuration file is not well-formed.")
        sys.exit(1)



# TODO: Method to parse data

##############################################################
# Set commandline paths in MTNN.cli_var
#############################################################
def set_script_path(script_path_arg: str):
    """
    Sets variable cli_var.SCRIPT_PATH
    Args:
        script_path_arg: <str> Absolute or relative path to the script

    Returns:
         None
    """
    env.SCRIPT_PATH = script_path_arg


def set_config_path(config_file_arg: str):
    """
    Sets variable cli_var.CONFIG_PATH.
    Args:
        config_file_arg: <str> Absolute or relative path to a file or a directory.

    Returns:
        None
    """
    env.CONFIG_PATH = config_file_arg

def set_dir_path(dir_path_arg: str):
    """
    Sets variables cli_var.DIR_PATH
    Args:
        dir_path_arg:

    Returns:
        None
    """
    env.DIR_PATH = dir_path_arg



#############################################################
# Set commandline flags  in mtnn_global
#############################################################
#TODO: Dead code. Refactored to env.py

def set_debug(debug_flag: bool):
    """
    Sets debug flag.
    Args:
        debug_flag: <bool> If true, generates logs from MTNN Model class to runs/ with extension .log.txt

    Returns: None
    """

    env.DEBUG = debug_flag


def set_logstdout(logstdout_flag : bool):
    """
    Sets debug flag.
    Args:
        logstdout_flag: <bool> If true, generates logs from MTNN Model class to runs/ with extension .stdout.txt

    Returns: Null
    """
    env.LOG_STDOUT = logstdout_flag
    if env.LOG_STDOUT is True:
        logger.set_fileout_name(env.CONFIG_PATH)
        sys.stdout = logger.StreamLogger()

