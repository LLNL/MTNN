"""
Logging set-up and formatting
"""
# standard
import sys
import logging

# local
import MTNN.utils.file as file


__all__ = ['progressbar',
           'get_logger']

# Set formatting
console_formatter = logging.Formatter('%(message)s')
file_formatter = logging.Formatter('%(message)s')

# File Handler
filepath = file.make_default_path("/logs/", ".txt")
fh = logging.FileHandler(filepath, mode = 'a')
fh.setLevel(level = logging.INFO)
fh.setFormatter(file_formatter)

# Console Handler
ch = logging.StreamHandler()  # writes to stderr
ch.setLevel(level=logging.INFO)
ch.setFormatter(console_formatter)


def progressbar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def get_logger(logger_name, write_to_file=False):
    """
    Set up logger for each module.
    Args:
        logger_name: Pass the module's __name__
        write_to_file : <bool>

    Returns:
        logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)  # Default

    if write_to_file:
        logger.addHandler(fh)

    logger.addHandler(ch)
    logger.propagate = False  # disable propagation to root logger

    return logger


