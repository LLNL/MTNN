#!/usr/bin/env python
""" MTNN/scripts/template_scripts/cli_find_overfit.py
Script to evaluate different MTNN model architectures using the MTNN core
to find one that overfits (where validation accuracy is greater than training accuracy)
* Reads configuration files from a directory
* Reads from any Pytorch torch.utils.data.Dataset
"""
# standard
import os
from pathlib import Path

# PyTorch

# local
import cli.env_var as cli # arguments from commandline TODO: cli_temporary.Variables -> logger.py

import MTNN as config_reader

###################################
# Load dataset
###################################
#data = datasets.load()

print(cli.DIR_PATH)
if not cli.DIR_PATH:
    for path in os.listdir(cli.DIR_PATH):
        config_file_path = Path(cli.DIR_PATH).joinpath(path).resolve()
        print(config_file_path)

        # Read in YAML file
        MyYamlConfig = config_reader.YamlConfig(config_file_path)

        # Create Model
        #model = builder.build_model(config_file_path, visualize=False, debug=cli_temporary.DEBUG)

        # Create Trainer
        #components = components.build_trainer(MyYamlConfig.components())



#####################################
# Train.
#####################################



#####################################
# Evaluate
#####################################



#####################################
# Plot
#####################################

