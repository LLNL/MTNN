"""
MTNN/methods.py
Static helper methods
"""
# standard
import sys
import os
import pathlib
import argparse


def parse_args() -> list:
    """
    Returns:
        commandline_args <list>
    """
    parser = argparse.ArgumentParser(description = "Runs MTNN test script with a YAML config file")
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
    # TODO: Option for directory of yaml files
    # TODO: Option for checkpointing
    # TODO: Option for tensorboard visualization
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
    print(cwd)
    results = []
    for root, dirs, files in os.walk(cwd):
        if filename in files:
            path = os.path.abspath(os.path.join(root, filename))
            results.append(path)
            return path
    if not results:
        print(f"Unable to find {filename} in current directory.")
        sys.exit(1)


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


