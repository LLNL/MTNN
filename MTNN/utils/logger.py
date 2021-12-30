import sys
import re
import datetime
from pathlib import Path, PurePath
import logging

__all__ = ['create_file_handler',
           'create_consoler_handler',
           'create_MTNN_logger',
           'get_MTNN_logger']

# Set formatting
console_formatter = logging.Formatter('%(message)s')
file_formatter = logging.Formatter('%(message)s')

MTNN_logger_name = ""

def make_path(dir: str, filename: str):
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
    default_file = sys.argv[0].strip(".py") + "_" \
                    + datetime.datetime.today().strftime("%m%d%Y") + "_" \
                    + datetime.datetime.now().strftime("%H:%M:%S") + "_" \
                    + datetime.datetime.today().strftime("%A") + ext
    default_path = default_dir.joinpath(Path(filename))

    basedir = PurePath(filepath).parent
    filename = PurePath(filepath).name

    # Create base directory
    if not Path(basedir).exists():
        Path(basedir).mkdir(parents=True)

    # Create file
    default_path = Path(basedir, filename)

    return default_path


def create_file_handler(filename, logging_level):
    # filepath = make_path(path, ".txt")
    # print("FILEPATH IS {}".format(filepath))
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(level = logging_level)
    fh.setFormatter(file_formatter)
    return fh

def create_console_handler(logging_level):
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging_level)
    ch.setFormatter(console_formatter)
    return ch
    
def create_MTNN_logger(logger_name, logging_level=logging.DEBUG, log_filename=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    if log_filename is not None:
        logger.addHandler(create_file_handler(log_filename, logging_level))

    logger.addHandler(create_console_handler(logging_level))

    global MTNN_logger_name
    MTNN_logger_name = logger_name
    return logger

def get_MTNN_logger():
    return logging.getLogger(MTNN_logger_name)
