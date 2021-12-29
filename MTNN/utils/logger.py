"""
Logging set-up and formatting
"""
# standard
import sys
import logging

__all__ = ['create_file_handler',
           'create_consoler_handler',
           'create_MTNN_logger',
           'get_MTNN_logger']

# Set formatting
console_formatter = logging.Formatter('%(message)s')
file_formatter = logging.Formatter('%(message)s')

MTNN_logger_name = ""

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


def create_file_handler(path, logging_level):
    filepath = make_default_path(path, ".txt.")
    fh = logging.FileHandler(filepath, mode='a')
    fh.setLevel(level = logging_level)
    fh.setFormatter(file_formatter)
    return fh

def create_console_handler(logging_level):
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging_level)
    ch.setFormatter(console_formatter)
    return ch
    
def create_MTNN_logger(logger_name, logging_level=logging.DEBUG, write_to_file=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    if write_to_file:
        logger.addHandler(create_file_handler("/logs", logging_level))

    logger.addHandler(create_console_handler(logging_level))

    global MTNN_logger_name
    MTNN_logger_name = logger_name
    return logger

def get_MTNN_logger():
    return logging.getLogger(MTNN_logger_name)
