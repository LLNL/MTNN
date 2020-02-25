"""
#Test prolongation operators
"""
# System packages
import pytest
import yaml
from itertools import permutations

# Pytorch packages
import torch
from torch import nn

# Local package
from MTNN import model as mf

# Instantiate Model and generate data to test on
@pytest.fixture
def my_model():
    my_model = mf.Model()
    return my_model


def generate_data(lambda_fn):
    tensor_data = []
    z = lambda_fn
    input_list = permutations(range(0,3), 2)
    for x, y in input_list:
        input = torch.FloatTensor([x, y])
        output = torch.FloatTensor([z(x, y)])
        tensor_data.append((input, output))
    return tensor_data


file = open("/Users/mao6/proj/mtnnpython/MTNN/tests/test.yaml", "r")
model_config = yaml.load(file, Loader = yaml.SafeLoader)

# Modify the lambda function
test_fn = lambda x,y: 3 * x + 2 *y
test_data = generate_data(test_fn)

##############################################################
# Tests.
###############################################################
def test_lowertriangular():
    pass
    # TODO: Test dimensions with different expansion factor
    # TODO: Check dimensions of output layer
    # TODO: Check distribution