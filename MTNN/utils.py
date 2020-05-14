import re
import os
import sys
import inspect
import logging
import datetime
from pathlib import Path, PurePath

import torch


def progressbar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


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
    filename_match = re.search("\w*.py$", caller_filepath)

    if type(filename_match) is not None:
        match = filename_match.group()
        clean_caller_filepath = caller_filepath.strip(match)
    return clean_caller_filepath


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
    #prev_caller_filename = pathlib.PurePath(caller_filepath)
    return prev_caller_filename


def get_logger(logger_name, create_file=False):
    """
    Set up logger for each module.
    Args:
        logger_name: Pass the module's __name__
        create_file : <bool>

    Returns:
        logger

    TODO: Log multiple modules to the same file (smoother, prolongation)

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)  # Default

    # Set formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    if create_file:
        filepath = make_default_path("/logs/", ".txt")

        fh = logging.FileHandler(filepath)
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def make_path(filepath):
    """
    Recursively makes a file path.
    Args:
        filepath: <string>

    Returns:
        ret_path: <Path> Posix or Windows file path object
    """
    basedir = PurePath(filepath).parent
    filename = PurePath(filepath).name

    # Create base directory
    if not Path(basedir).exists():
        Path(basedir).mkdir(parents=True)

    # Create file
    ret_path = Path(basedir, filename)

    return ret_path


def make_default_path(dir: str, ext: str):
    """
    Creates a sub directory if it doesn't exist and default filename based on caller filename.
        <filename>_MMDDYYY_HH:MM:SS_DayofWeek_<ext>
    Returns open file object.
    Args:
        dir: <str> Name of the sub directory.
        ext: <str> Name of the file extension

    Returns:
        default_path <Path> Posix or Windows file path object
    """
    dirname = re.sub('[\\\/]', '', dir)
    default_dir = PurePath.joinpath(Path.cwd(), Path("./" + dirname))
    default_file = get_caller_filename() + "_" \
                    + datetime.datetime.today().strftime("%m%d%Y") + "_" \
                    + datetime.datetime.now().strftime("%H:%M:%S") + "_" \
                    + datetime.datetime.today().strftime("%A") + ext
    default_path = default_dir.joinpath(Path(default_file))

    default_path = make_path(default_path)

    return default_path


def enable_cuda():
    # TODO
    pass


def plot():
    #TODO
    pass