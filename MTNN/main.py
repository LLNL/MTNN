# !/usr/bin/env/ python
"""
Calls a script to build and run a neural network with the specified configuration file.
"""

# standard
import os

# local source
from MTNN import methods
from MTNN import logger
from MTNN import mtnn_var


if __name__ == "__main__":
    # Parse command line arguments
    args = methods.parse_args()

    script_path_arg = args.script[0]
    config_path_arg = args.configuration[0]
    # TODO: Add debug option

    # Set MTNN Global variables
    # Script path
    mtnn_var.set_script_path(methods.check_path(script_path_arg))

    # Validate YAML configuration file
    methods.check_config(config_path_arg) #TODO: fill in

    # Configuration file path
    # Absolute/relative path given
    if os.path.exists(config_path_arg):
        mtnn_var.set_config_path = methods.check_path(config_path_arg)

    # Filename given
    else:
        config_dir = os.path.dirname(mtnn_var.SCRIPT_PATH)
        config_path = methods.find_config(config_dir, config_path_arg)
        mtnn_var.set_config_path(config_path)

    # Set logger
    logger.set_fileout_name(mtnn_var.CONFIG_PATH)

    # Execute the script
    with open(mtnn_var.SCRIPT_PATH) as f:
        code = compile(f.read(), mtnn_var.SCRIPT_PATH, 'exec')
        exec(code, locals())


