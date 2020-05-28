import re
import os
import inspect
import datetime
from pathlib import Path, PurePath

__all__ =['get_caller_filename',
          'get_caller_filepath',
          'make_path',
          'make_default_path']



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