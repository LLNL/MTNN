# !/usr/bin/env/ python
"""
Calls a script to build and run a neural network with the specified configuration file.
"""

# standard
import os

# local source
from MTNN import mtnn_utils
from MTNN import logger
from MTNN import mtnn_var


if __name__ == "__main__":
    # Parse command line arguments
    args = mtnn_utils.parse_args()

    script_path_arg = args.script[0]
    config_path_arg = args.configuration[0]
    # TODO: Add debug option

    # Set MTNN Global variables
    # Script path
    mtnn_var.set_script_path(mtnn_utils.check_path(script_path_arg))

    # Validate YAML configuration file
    mtnn_utils.check_config(config_path_arg) #TODO: fill in

    # Configuration file path
    # Given Absolute/relative path given
    if os.path.exists(config_path_arg):
        clean_config_path = mtnn_utils.check_path(config_path_arg)
        mtnn_var.set_config_path(clean_config_path)

    # Given Filename only
    else:
        config_dir = os.path.dirname(mtnn_var.SCRIPT_PATH)
        config_path = mtnn_utils.find_config(config_dir, config_path_arg)
        mtnn_var.set_config_path(config_path)

    # Set logger
    logger.set_fileout_name(mtnn_var.CONFIG_PATH)

    # Execute the script
    with open(mtnn_var.SCRIPT_PATH) as f:
        code = compile(f.read(), mtnn_var.SCRIPT_PATH, 'exec')
        exec(code, locals())


