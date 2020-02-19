"""
Global variables set from main
"""
SCRIPT_PATH = ""
CONFIG_PATH = ""


# Mutator methods.
def set_script_path(script_path: str):
    global SCRIPT_PATH
    SCRIPT_PATH = script_path


def set_config_path(config_file: str):
    global CONFIG_PATH
    CONFIG_PATH = config_file

