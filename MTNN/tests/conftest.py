"""
 Filename: MTNN/tests/conftest.py
 PyTest Hooks for global set-up and tear-down for MTNN/tests
"""

# standard packages
import os
import logging

# third-party packages
import pytest
import yaml
import pprint

# pytorch
import torch
from torch.autograd import Variable
import sklearn.datasets as skdata

# local package
from core import model as mtnnmodel
from scratches.codebase import constants
from configuration import generator

# Logging set-up
# TODO: Configure Logger
# TODO: Clean print statements -> logger

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
pp = pprint.PrettyPrinter(indent=4)

##################################################
# Set-up code.
##################################################

# Generate YAML configuration files.
@pytest.fixture(autouse=False, scope='session') # TODO: Change to True
def gen_configs():
    """
    Generates YAML configuration files.
    Returns:
        config_dir <string> Absolute directory path where generated YAML configuration files are stored.
    """
    print("\nSETUP: Generating configuration files ")
    if generator.dir_is_empty():
        pass
    else:
        generator.main()

    # Get configuration files directory path.
    print("\nSETUP: Config files stored in " + generator.get_config_dir())

    return


# Generate MTNN Models.
@pytest.yield_fixture(autouse=True, scope='session')
def make_models():
    """
    Returns an iterator that yields a MTNN.Model object
    Args:
        gen_configs <string> YAML configuration file directory
    Returns:
        model <MTNN.Model> Instance of a MTNN.Model object
    """
    print("\nSETUP: Collection_of_models")

    collection_of_models = []
    for yaml_file in os.listdir(constants.POSITIVE_TEST_DIR):

        yaml_file_path = constants.POSITIVE_TEST_DIR + "/" + yaml_file

        config = yaml.load(open(yaml_file_path, 'r'), Loader = yaml.SafeLoader)
        model = mtnnmodel.Model(config)
        model.set_config()
        collection_of_models.append(model)

    # Logging
    logging.debug(collection_of_models)

    yield collection_of_models


# TODO: Test regression_training_data on test_model/test_prolongation

# Generate training Data.
@pytest.yield_fixture(autouse=True, scope='session')
def regression_training_data():
    """
    Returns an iterator that yields a tuple of tensor training input and output data
    from a randomly generated regression problem. To change test problem parameters,
    modify TEST_FN_PARAMETERS in tests_var.py
    Returns:
        training_data_input <tensor>
        training_data_output <tensor>

    """
    print("\nSETUP: Generating regression training data")
    x, y = skdata.make_regression(n_samples= constants.TEST_FN_PARAMETERS['n_samples'],
                                  n_features= constants.TEST_FN_PARAMETERS['n_features'],
                                  noise= constants.TEST_FN_PARAMETERS['noise'])
    # Reshape.
    y.shape = x.shape

    # Tensorize data.
    for i in range(len(x)):
        training_data_input = Variable(torch.FloatTensor(x), requires_grad = True)
        training_data_output = Variable(torch.FloatTensor(x))

    # Logging
    logging.debug("Regression Training Input Data")
    logging.debug(training_data_input)
    logging.debug("Regression Training Output Data")
    logging.debug(training_data_output)

    yield training_data_input, training_data_output


##################################################
# Teardown code.
##################################################
@pytest.fixture(autouse=True, scope="session")
def teardown():
    print("TEARDOWN")
    # TODO: Fill in teardown code


##################################################
# Global Test Reports
##################################################
# TODO: Post-process test reports/failures

# TODO: Configuration Header


