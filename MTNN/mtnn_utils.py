"""
MTNN/mtnn_utils.py
Static helper methods
"""
# standard
import sys
import os
import pathlib
import argparse
import inspect
import re


def parse_args() -> list:
    """
    Returns:
        commandline_args <list>
    """
    parser = argparse.ArgumentParser(description = "Runs MTNN script with a YAML config file")
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
    # Option for debugging
    parser.add_argument("-d", "--debug",
                        default = sys.stdin,
                        action = "store_true",
                        help = "Sets the flag for debug logs")
    # TODO: Option for directory of yaml files
    # TODO: Option for checkpointing
    # TODO: Option for tensorboard visualization

    # Parse commandline arguments
    commandline_args = parser.parse_args()

    return commandline_args


def find_config(config_dir: str, filename: str) -> str:
    """
    Searches the base directory given by config_dir for the filename.
    Args:
        config_dir: <str> Directory path
        filename: <str> Name of the configuration file

    Returns:
        path: <str> Absolute UNIX path

    """
    cwd = os.path.dirname(config_dir)
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


# TODO: Fill in. Validate YAML config file. Use Python confuse API?
def check_config(filepath: str) -> bool:
    """
    Check if config file can be read without errors, is not empty and is well-formed YAML file
    Args:
        filepath (str) Full path to the YAML configuration file
    Returns: bool
    """
    try:
        pass
    except ValueError:
        print("Configuration file is not well-formed.")
        sys.exit(1)


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

    if type(filename_match) is not None:
        match = filename_match.group()
        clean_caller_filepath = (caller_filepath).strip(match)
    return clean_caller_filepath


