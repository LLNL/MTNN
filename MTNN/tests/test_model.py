import pytest

import torch
import MTNN.core.components.models as models
from MTNN.core.optimizer.operators import prolongation



@pytest.fixture
# Fixtures act as function arguments
def model():
    my_model = models.MultiLinearNet([10, 10, 1])
    return my_model


class TestModel:
    def test_model(self, model):
        print(model.layers)

    def test_model_weights(self, model):
        """
        Tests for weights using fixed seeds for Torch's random number generator.
        """
        for seed in range(10):

            # Set for deterministic results
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True



            # Apply prolongation operator
            for expansion_factor in range(10):
                low_tri_op = prolongation.LowerTriangleProlongation(expansion_factor)
                prolonged_model = low_tri_op.apply(source_model = model)

                error_msg = "Prolonged copied weights are incorrect."

                """
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
                                    assert torch.all(
                                        torch.eq(p_mod_layer.weight.data[0], o_mod_layer.weight.data[row])), \
                                        error_msg

                            # Middle hidden layers
                            elif 0 < module_index < (len(prolonged_model._module_layers) - 1):
                                for row in range(o_mod_layer.weight.size()[0]):
                                    for p_element, o_element in zip(p_mod_layer.weight.data[row],
                                                                    o_mod_layer.weight.data[row]):
                                        assert torch.all(torch.eq(p_element, o_element)), \
                                            error_msg

                            # Last hidden layer
                            else:
                                for p_element, o_element in zip(p_mod_layer.weight.data[-1],
                                                                o_mod_layer.weight.data[-1]):
                                    assert torch.all(torch.eq(p_element, o_element)), error_msg
                                pass
                 """

