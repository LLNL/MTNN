"""
Logs stdout to file while still printing to console
"""
# standard
import sys
import os
import datetime

# local source
from scratches.codebase import constants
from MTNN import cli

# TODO: Clean


# Called from Main

def set_fileout_name(config_path: str) -> str:
    """
    Formats the log filename given by config_path. Example filename:

    Args:
        config_path: <str>

    Returns:
       None

    """
    config_base_name = os.path.basename(config_path)
    cli.env_var.LOGFILE_NAME = constants.EXPERIMENT_LOGS_DIR \
                               + "/" + utils.get_caller_filename() + "_" \
                               + config_base_name + "_" \
                               + datetime.datetime.today().strftime("%m%d%Y") + "_" \
                               + datetime.datetime.now().strftime("%H:%M:%S") + "_" \
                               + datetime.datetime.today().strftime("%A") \
                               + ".stdout.txt"


class StreamLogger:
    """
    Logs stdout to the file path given by FILEOUT
    """
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(cli.env_var.LOGFILE_NAME, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

