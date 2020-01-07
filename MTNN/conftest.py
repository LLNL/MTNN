# MTNN/tests/conftest.py
# PyTest plug-in
# Hooks for global set-up and tear-down functions for MTNN/tests

# Used to import external plug-ins or modules
# Declare directory-specific hooks/fixtures such as set-up and tear-down methods
# Place conftest.py in root dir/path s.t. pytest recognizes application modules without specifying PYTHONPATH

# Built-in packages
import os

# External packages
import pytest
import yaml
import torch
from torch.autograd import Variable
import sklearn.datasets as data
import numpy as np

# Local package
from MTNN import model as mtnnModel
from MTNN import config_generator
from MTNN import TESTFN_PARAMS

##################################################
# Set-up code.
##################################################

# Generate YAML configuration files.
@pytest.fixture(autouse=True, scope='session')
def gen_yamlconfigs():
    """
    Generates YAML configuration files and returns the directory path where they are stored.
    Returns:
        config_dir <string> Absolute directory path where generated YAML configuration files are stored.
    """

    if config_generator.dir_is_empty():
        pass
    else:
        config_generator.main()

    # Get configuration files directory path.
    config_dir = config_generator.get_config_dir()
    print("\nSETUP: Generated config files to " + config_dir)

    return config_dir


# Model Object Factory.
@pytest.fixture(autouse=True, scope='session')
def make_models(gen_yamlconfigs):
    """
    Generator function: yields a MTNN.Model object
    Args:
        gen_configs <string> YAML configuration file directory
    Returns:
        model <MTNN.Model> Instance of a MTNN.Model
    """
    print("\nSETUP: Collection_of_models")

    for yaml_file in os.listdir(gen_yamlconfigs):

        config = yaml.load(open(gen_yamlconfigs + "/" + yaml_file, 'r'), Loader = yaml.SafeLoader)
        model = mtnnModel.Model(config)

    yield model


# TODO: Generate Dummy Training Data.



##################################################
# Teardown code.
##################################################
@pytest.fixture(scope="session")
def teardown():
    print("TEARDOWN run_tests")
    # TODO: Fill in teardown code



##################################################
# Global Test Reports
##################################################
# TODO: Post-process test reports/failures

# TODO: Configuration Header


