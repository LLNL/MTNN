"""
 Filename: MTNN/tests/conftest.py
 PyTest Hooks for global set-up and tear-down functions for MTNN/tests
"""
# Standard packages
import os
import logging

# Third-party packages
import pytest
import yaml
import pprint

# Pytorch
import torch
from torch.autograd import Variable
import sklearn.datasets as skdata

# Local package
import globalvar as gv
from MTNN import model as mtnnmodel
from MTNN import config_generator


# Logging set-up
# TODO: Configure Logger
# TODO: Clean print statements -> logger
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
pp = pprint.PrettyPrinter(indent=4)

##################################################
# Set-up code.
##################################################

# Generate YAML configuration files.
@pytest.fixture(autouse=True, scope='session')
def gen_configs():
    """
    Generates YAML configuration files.
    Returns:
        config_dir <string> Absolute directory path where generated YAML configuration files are stored.
    """
    print("\nSETUP: Generating config files ")
    if config_generator.dir_is_empty():
        pass
    else:
        config_generator.main()

    # Get configuration files directory path.
    print("\nSETUP: Config files stored in " + config_generator.get_config_dir())

    return


# Generate MTNN Models.
@pytest.yield_fixture(autouse=True, scope='session')
def make_models():
    """
    Returns an iterator that yields a MTNN.Model object
    Args:
        gen_configs <string> YAML configuration file directory
    Returns:
        model <MTNN.Model> Instance of a MTNN.Model
    """
    print("\nSETUP: Collection_of_models")

    collection_of_models = []
    for yaml_file in os.listdir(gv.POSITIVE_TEST_DIR):

        yaml_file_path = gv.POSITIVE_TEST_DIR + "/" + yaml_file

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
    modify TEST_FN_PARAMETERS in MTNN/__init__.py
    Returns:
        training_data_input <tensor>
        training_data_output <tensor>

    """
    print("\nSETUP: Generating regression training data")
    x, y = skdata.make_regression(n_samples=gv.TEST_FN_PARAMETERS['n_samples'],
                                  n_features=gv.TEST_FN_PARAMETERS['n_features'],
                                  noise=gv.TEST_FN_PARAMETERS['noise'])
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


