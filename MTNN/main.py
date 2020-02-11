# !/usr/bin/env/ python
"""
Calls a script to build and run a neural network with the specified configuration file.
"""

# standard library packages
import sys
import argparse
import os
import pathlib
from os import path


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


def check_paths(filepath: str) -> str:
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
    if not path.isabs(filepath):
        clean_filepath = filepath.strip("./")
        abs_filepath = path.join(path.abspath('.'), clean_filepath)

    else:
        abs_filepath = filepath
    return abs_filepath


def check_config(filepath: str) -> bool:
    """
    Check if config file can be read without errors, is not empty and is well-formed YAML file
    Args:
        filepath (str) Full path to the YAML configuration file
    Returns: bool
    """
    # TODO: Validate YAML config file. Use Python confuse API?
    try:
        pass
    except ValueError:
        print("Configuration file is not well-formed.")
        sys.exit(1)



if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    script_path = check_paths(args.script[0])
    config_path = check_paths(args.configuration[0])

    # Validate YAML configuration file
    check_config(config_path)

    # Execute the script
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, locals())


