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
from core import build, model as mtnnmodel
from core.components.trainer.optimizer import prolongation
from scratches.codebase import trainer, constants

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

CONFIG_DIR = constants.POSITIVE_TEST_DIR

@pytest.fixture
def my_model():
    my_model = mtnnmodel.Model()
    return my_model


def test_config(my_model):
    """
    Test if model attributes are set correctly after building from configuration file.
    Runs on configuration files generated by conftest.gen_configs() given by the variable CONFIG_DIR
    """

    for config_file in os.listdir(CONFIG_DIR):

        # Build Model
        config_file_path = constants.POSITIVE_TEST_DIR + "/" + config_file
        print("\n\nCONFIGURATION FILE", config_file_path)
        my_model = build.build_model(config_file_path)

        # Test attributes
        assert len(my_model._model_type) != 0
        assert my_model._input_size > 0
        assert my_model._layer_config is not None
        assert len(my_model._module_layers) != 0
        assert issubclass(type(my_model._objective_fn.__class__), type(nn.Module))
        assert my_model._hyperparameters is not None

        # Set Optimizer
        optimizer = trainer.build_optimizer(config_file_path, my_model.parameters())
        my_model.set_optimizer(optimizer)
        assert my_model._objective_fn != None
        assert my_model._optimizer != None


def test_prolongation_lowertriangular(my_model):
    """
    Tests for model after applying LowerTriangular prolongation operator.
    Runs on all configuration files given by the variable CONFIG_DIR.
    """
    # TODO: test with multiple expansion factors
    for config_file in os.listdir(CONFIG_DIR):
        # Build Model
        config_file_path = constants.POSITIVE_TEST_DIR + "/" + config_file
        print("\n\nCONFIGURATION FILE", config_file_path)
        my_model = build.build_model(config_file_path)

        # Set Optimizer
        optimizer = trainer.build_optimizer(config_file_path, my_model.parameters())
        my_model.set_optimizer(optimizer)

        # Apply prolongation operator
        expansion_factor = 3 # Fix this.
        lowtri_op = prolongation.LowerTriangleOperator()
        prolonged_model = lowtri_op.apply(source_model=my_model, exp_factor=expansion_factor)

        prolonged_model.print_properties()

        # Check class
        assert type(prolonged_model) == type(my_model)

        # Check attributes
        assert my_model._layer_config != None
        assert prolonged_model._model_type == my_model._model_type
        assert prolonged_model.num_layers == my_model.num_layers
        assert prolonged_model._input_size == my_model._input_size
        assert prolonged_model._layer_config is not None
        assert len(prolonged_model._module_layers) == len(my_model._module_layers)
        assert prolonged_model._hyperparameters == my_model._hyperparameters

        # Check hyperparameters
        assert type(prolonged_model._objective_fn) == type(my_model._objective_fn)
        assert type(prolonged_model._optimizer) == type(my_model._optimizer)

        # Check weights
        print("\n\nCHECKING WEIGHTS, BIASES, AND DIMENSIONS...")
        for module_index in range(len(prolonged_model._module_layers)):
            mod_key = 'layer' + str(module_index)

            original_model_modlayers = my_model._module_layers[mod_key]
            prolonged_model_modlayers = prolonged_model._module_layers[mod_key]

            prev_p_mod_layer_outsize = 0

            for (p_mod_layer, o_mod_layer) in zip(prolonged_model_modlayers, original_model_modlayers):

                # Linear layers only
                if hasattr(p_mod_layer, "weight") and hasattr(o_mod_layer, "weight"):
                    # My_model and prolong_model dimensions

                    p_mod_layer_numrow = p_mod_layer.weight.size()[0]
                    p_mod_layer_numcol = p_mod_layer.weight.size()[1]
                    o_mod_layer_numrow = o_mod_layer.weight.size()[0]
                    o_mod_layer_numcol = o_mod_layer.weight.size()[1]

                    # First hidden layer
                    if module_index == 0:
                        # Check Values
                        assert torch.all(torch.eq(p_mod_layer.weight.data[0], o_mod_layer.weight.data[0])), \
                            "Weights were copied incorrectly"
                        # TODO: Check random weights and distribution
                        """
                        for row in p_mod_layer.weight.data:
                            print(1/math.sqrt(expansion_factor * p_mod_layer_numcol))
                            assert max(row) <= 1/math.sqrt(expansion_factor * p_mod_layer_numcol)  # Check this. 
                            assert (min(row)) >= -(1/math.sqrt(expansion_factor * p_mod_layer_numcol)) #Check this. 
                        """
                        # Check Dimensions
                        assert p_mod_layer_numrow == expansion_factor * o_mod_layer_numrow,\
                            "Prolonged row dimensions are incorrect."
                        assert p_mod_layer_numcol == o_mod_layer_numcol,\
                            "Prolonged column dimensions are incorrect."
                        prev_p_mod_layer_outsize = p_mod_layer_numrow #TODO: FIX

                    # Middle hidden layers
                    elif 0 < module_index < (len(prolonged_model._module_layers) - 1):

                        # TODO: Check random weights distribution
                        # TODO: Check size of zero block
                        # TODO: Check lower triangular block shape

                        # Dimensions
                        assert p_mod_layer_numrow == expansion_factor * o_mod_layer_numrow,\
                            "Prolonged row dimensions are incorrect."
                        assert p_mod_layer_numcol == prev_p_mod_layer_outsize,\
                            "Prolonged column dimensions are incorrect."

                    # Last hidden layer
                    else:
                        assert p_mod_layer_numrow == o_mod_layer_numrow,\
                            "Prolonged row dimensions are incorrect."
                        assert p_mod_layer_numcol == expansion_factor * o_mod_layer_numcol,\
                            "Prolonged column dimensions are incorrect."


def test_model_weights(my_model):
    """
    Tests for weights using fixed seeds for Torch's random number generator.
    """
    for s in range(1000):

        # Set for deterministic results
        torch.manual_seed(s)
        torch.backends.cudnn.deterministic = True

        for config_file in os.listdir(CONFIG_DIR):
            # Build Model
            config_file_path = constants.POSITIVE_TEST_DIR + "/" + config_file
            print(f"\nCONFIGURATION FILE: {config_file_path}")
            print(f"USING TORCH SEED:{s}", torch.rand(1))
            my_model = build.build_model(config_file_path)

            # Set Optimizer
            optimizer = trainer.build_optimizer(config_file_path, my_model.parameters())
            my_model.set_optimizer(optimizer)

            # Apply prolongation operator
            expansion_factor = 3 # Fix this.
            low_tri_op = prolongation.LowerTriangleOperator()
            prolonged_model = low_tri_op.apply(source_model=my_model, exp_factor=expansion_factor)

            error_msg = "Prolonged copied weights are incorrect."
            for module_index in range(len(prolonged_model._module_layers)):
                mod_key = 'layer' + str(module_index)

                original_model_modlayers = my_model._module_layers[mod_key]
                prolonged_model_modlayers = prolonged_model._module_layers[mod_key]

                for (p_mod_layer, o_mod_layer) in zip(prolonged_model_modlayers, original_model_modlayers):
                    # Linear layers only
                    if hasattr(p_mod_layer, "weight") and hasattr(o_mod_layer, "weight"):

                        # First hidden layer
                        if module_index == 0:
                            for row in range(o_mod_layer.weight.size()[0]):
                                assert torch.all(torch.eq(p_mod_layer.weight.data[0], o_mod_layer.weight.data[row])),\
                                    error_msg

                        # Middle hidden layers
                        elif 0 < module_index < (len(prolonged_model._module_layers) - 1):
                            for row in range(o_mod_layer.weight.size()[0]):
                                for p_element, o_element in zip(p_mod_layer.weight.data[row],
                                                                o_mod_layer.weight.data[row]):
                                    assert torch.all(torch.eq(p_element, o_element)),\
                                        error_msg

                        # Last hidden layer
                        else:
                            for p_element, o_element in zip(p_mod_layer.weight.data[-1], o_mod_layer.weight.data[-1]):
                                assert torch.all(torch.eq(p_element, o_element)), error_msg
                            pass


def test_input(regression_training_data):
    """"
    (model_input, expected_output) = regression_training_data
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