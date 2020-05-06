import pathlib
import re
import os
import sys
import inspect
import logging
from pathlib import PurePath, Path

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
        create_file:

    Returns:

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)  # Default

    # Formatter
    formatter = logging.Formatter()

    #File Handler
    if create_file:

        filepath = get_caller_filepath()  + "/log" + get_caller_filename() + ".log"
        print(filepath)
        fh = logging.FileHandler(get_caller_filename() + ".log")
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger



def make_path(dir, filename):
    try:
        # Check exists
        if not Path(dir).exists():
            Path(dir).mkdir()

        if not Path(filename).exists():
            cleanpath = Path(filename).resolve()
            print(cleanpath)

    except IOError or FileNotFoundError as exc:
        print(exc)

    return cleanpath



def plot():
    #TODO: Fill in
    pass