"""MTNN/logger.py
Logs stdout to file while still printing to console
"""
# standard
import sys
import os
import datetime

# local source
import MTNN.mtnn_defaults as mtnn_defaults
import MTNN.mtnn_utils as mtnn_utils

# TODO: Clean


# Called from Main
def set_fileout_name(config_path: str) -> str:
    """
    Formats the log filename given by config_path.
    Args:
        config_path: <str>

    Returns:
        FILEOUT: <str>

    """
    config_base_name = os.path.basename(config_path)
    global FILEOUT
    FILEOUT = mtnn_defaults.EXPERIMENT_LOGS_DIR \
              + "/" + mtnn_utils.get_caller_filename() + "_" \
              + config_base_name + "_" \
              + datetime.datetime.today().strftime("%m%d%Y") + "_" \
              + datetime.datetime.now().strftime("%H:%M:%S") + "_" \
              + datetime.datetime.today().strftime("%A") \
              + ".stdout.txt"
    return FILEOUT


class StreamLogger:
    """
    Logs stdout to the file path given by FILEOUT
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(FILEOUT, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

