"""

"""
# standard
import inspect
import os
import re


@staticmethod
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


@staticmethod
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