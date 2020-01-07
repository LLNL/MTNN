# MTNN/tests/conftest.py
# PyTest Hooks for global set-up and tear-down functions for MTNN/tests

# Built-in packages
import os

# External packages
import pytest
import yaml
import torch
from torch.autograd import Variable
import sklearn.datasets as data


# Local package
from MTNN import model as mtnnModel
from MTNN import config_generator
from MTNN import TEST_FN_PARAMETERS


##################################################
# Set-up code.
##################################################

# Generate YAML configuration files.
@pytest.fixture(autouse=True, scope='session')
def gen_configs():
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
def make_models(gen_configs):
    """
    Generator function: yields a MTNN.Model object
    Args:
        gen_configs <string> YAML configuration file directory
    Returns:
        model <MTNN.Model> Instance of a MTNN.Model
    """
    print("\nSETUP: Collection_of_models")

    for yaml_file in os.listdir(gen_configs):

        config = yaml.load(open(gen_configs + "/" + yaml_file, 'r'), Loader = yaml.SafeLoader)
        model = mtnnModel.Model(config)

    yield model


# TODO: Test regression_training_data on test_model/test_prolongation

@pytest.fixture(autouse=True, scope='session')
def regression_training_data():
    """
    Generator function: yields tensor training (input, output) data from a randomly generated regression problem.
    To change test problem parameters, modify TEST_FN_PARAMETERS in MTNN/__init__.py
    Returns:
        training_data_input <tensor>
        training_data_output <tensor>

    """
    print("\nSETUP: Generating regression training data")
    x, y = data.make_regression(n_samples = TEST_FN_PARAMETERS['n_samples'],
                                n_features = TEST_FN_PARAMETERS['n_features'],
                                noise = TEST_FN_PARAMETERS['noise'])
    # Reshape.
    y.shape = x.shape

    # Tensorize data.
    for i in range(len(x)):
        training_data_input = Variable(torch.FloatTensor(x), requires_grad = True)
        training_data_output = Variable(torch.FloatTensor(x))

    yield training_data_input, training_data_output


##################################################
# Teardown code.
##################################################
@pytest.fixture(autouse=True, scope="session")
def teardown():
    print("TEARDOWN run_tests")
    # TODO: Fill in teardown code


##################################################
# Global Test Reports
##################################################
# TODO: Post-process test reports/failures

# TODO: Configuration Header


