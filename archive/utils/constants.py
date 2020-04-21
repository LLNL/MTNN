"""
constants.py
* filenames for logs
"""
# standard
import os
import datetime

# local
import MTNN.utils as utils

LOGS_DIR = os.path.abspath(os.path.join(utils.filehandler.get_caller_filepath() + "runs/logs/"))

if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

LOGS_FILENAME = os.path.join(LOGS_DIR + "/" + utils.filehandler.get_caller_filename() + "_"
                             + datetime.datetime.today().strftime("%m%d%Y") + "_"
                             + datetime.datetime.now().strftime("%H:%M:%S") + "_"
                             + datetime.datetime.today().strftime("%A"))
