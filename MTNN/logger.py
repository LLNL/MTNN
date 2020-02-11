"""MTNN/logger.py
Logs stdout to file while still printing to console
"""
# standard
import sys
import datetime

# local source
from MTNN import mtnn_defaults

FILEOUT = mtnn_defaults.EXPERIMENT_LOGS_DIR\
          + "/" + mtnn_defaults.get_caller_filename() + "_"\
          + datetime.datetime.today().strftime("%m%d%Y") + "_"\
          + datetime.datetime.now().strftime("%H:%M:%S") + "_"\
          + datetime.datetime.today().strftime("%A") + "_" \
          + ".stdout.txt"

class StreamLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(FILEOUT, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

