# Filename: MTNN/tests/test_model.py
# Baseline Unit test for MTNN.model on a fully connected network

# Built-in packages
import os

# External packages
import pytest

# Pytorch packages
import torch
from torch import nn

# Local package
from MTNN import model as mtnnmodel
from MTNN import CONFIG_DIR

# TODO: Generate tests with diff layer/neuron combinations

"""
def generate_data(lambda_fn):
    tensor_data = []
    z = lambda_fn
    input_list = permutations(range(0,3), 2)
    for x, y in input_list:
        input = Variable(torch.FloatTensor([x,y]), requires_grad = True)
        output = Variable(torch.FloatTensor([z(x,y)]))
        tensor_data.append((input, output))
    return tensor_data

file = open("/Users/mao6/proj/mtnnpython/MTNN/tests/test.yaml", "r")
model_config = yaml.load(file, Loader = yaml.SafeLoader)

test_fn = lambda x,y: 3 * x + 2 *y
test_data = generate_data(test_fn)
"""

@pytest.fixture
def my_model():
    my_model = mtnnmodel.Model()
    return my_model


def test_set_config(my_model, gen_configs):
    """Test model attributes set from config file"""

    for config_file in os.listdir(gen_configs):
        my_model.set_config(CONFIG_DIR + "/" + config_file)

        assert len(my_model._model_type) != 0
        assert my_model._input_size > 0
        assert my_model._num_layers != 0
        assert len(my_model._layers) == my_model._num_layers
        assert issubclass(type(my_model._objective_fn.__class__), type(nn.Module))
        # TODO: Better test for loss function is subclass of nn.modules.loss
        # TODO: Test for the Optimizer


def test_input(regression_training_data):
    """"
    (model_input, expected_output) = data
    my_model.set_config(model_config)
    assert len(model_input) == my_model.input_size
    """

# Test model sizes weights and biases
def test_parameters(my_model):
    #TODO
    pass

def test_fit (my_model):
    # TODO
    # assert loss type
    pass

def test_forward(my_model):
    #TODO
    pass

def test_gradients(my_model):
    # TODO
    # Use autograd.gradcheck
    # autograd.profiler
    # autograd.anomaly detection
    pass


def test_checkpoint():
    # TODO
    pass

def test_validate():
    # TODO
    pass

# TODO: test target size
