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

MTNN_logger_name = ""

def create_file_handler(path, logging_level):
    filepath = file.make_default_path(path, ".txt.")
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

def get_logger(a1, write_to_file):
    return get_MTNN_logger()
