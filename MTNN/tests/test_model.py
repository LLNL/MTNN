#Test fully connected neural network model
# System packages
import pytest
import yaml
from itertools import permutations

# Pytorch packages
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import gradcheck

# Local package
from MTNN import model as mf


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


@pytest.fixture
def my_model():
    my_model = mf.Model()
    return my_model


def test_set_config(my_model):
    my_model.set_config(model_config)
    assert len(my_model.model_type) != 0
    assert my_model.input_size > 0
    assert my_model.layer_num != 0
    assert len(my_model.layers) == my_model.layer_num


@pytest.mark.parametrize("data", test_data)
def test_input(my_model, data):
    (input, expected_output) = data
    my_model.set_config(model_config)
    assert len(input) == my_model.input_size


def test_hyperparameters(my_model):
    # TODO
    pass

# Test model weights and biases
def test_parameters(my_model):
    #TODO
    pass

def test_fit (my_model):
    # TODO
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





