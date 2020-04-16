# !/usr/bin/env_var/ python
""" MTNN/run.py
Commandline tool to call a script to builder and run a neural network with the specified configuration file.
"""

# standard
import subprocess
from pathlib import Path, PurePath

# local source
from logging import logger
import MTNN.cli.utils as utils
# import MTNN.commandline.env_var as env_var

if __name__ == "__main__":
    #######################################
    # Parse command line arguments
    #######################################
    args = utils.parse_args()

    # Paths
    arg_script_path = args.script[0]
    arg_config_path = args.configuration[0]

    # Flags
    arg_debug = args.debug
    arg_logstdout = args.log

    #######################################
    # Set MTNN Global variables
    ######################################

    # Script path
    clean_script_path = utils.check_path(arg_script_path)
    utils.set_script_path(clean_script_path)

    # Validate YAML configuration file
    utils.check_config(arg_config_path) #TODO: fill in

    # Configuration file path
    confPath = Path(arg_config_path)

    # Given Absolute/relative path of single file
    if confPath.exists() and confPath.is_file():
        clean_config_path = utils.check_path(confPath)
        utils.set_config_path(clean_config_path)

    # Given path to directory of files
    elif confPath.exists() and confPath.is_dir():
        # TODO: utils.check_dir_path
        utils.set_dir_path(confPath)

    # Gvien filename only
    else:
        config_dir = PurePath(env_var.SCRIPT_PATH).parent
        config_path = utils.find_config(config_dir, arg_config_path)
        utils.set_config_path(config_path)

    # Set flags
    utils.set_debug(arg_debug)
    utils.set_logstdout(arg_logstdout)

    ######################################
    # Set up logger
    ######################################
    # Log Stdout
    logger.set_fileout_name(env_var.CONFIG_PATH)

    ######################################
    # Execute the script
    #####################################
    """
    with open(cli_temporary.SCRIPT_PATH) as f:
        # TODO: Refactor with Subprocess module
        code = compile(f.read(), cli_temporary.SCRIPT_PATH, 'exec')
        exec(code)
    """

    subprocess.call([cli.env_var.SCRIPT_PATH])

