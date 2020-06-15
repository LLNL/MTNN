import pytest

import torch
import MTNN.core.components.models as models
from MTNN.core.multigrid.operators import prolongation



@pytest.fixture # Fixtures act as function arguments
def model():
    my_model = models.MultiLinearNet([2, 2, 1, 1])
    return my_model


class TestModel:
    def test_model(self, model):
        #print(model.layers)
        pass

    def test_dim(self, model):
        """
        Test weight and bias dimensions are correct
        """

        for expansion_factor in range(1, 3):
            print("\nExpansion Factor", expansion_factor)
            low_tri_op = prolongation.LowerTriangleProlongation(expansion_factor)
            prolonged_model = low_tri_op.apply(source_model = model, verbose=True)

            error_msg = "Prolonged copied weights are incorrect."

            print(model.layers)
            print(prolonged_model.layers)

            for layer_idx, (s_layer, p_layer) in enumerate(zip(model.layers, prolonged_model.layers)):

                if layer_idx == 0:
                    #print(layer.weight.size())
                    #print(layer.weight.size()[1])

                    print(p_layer.weight.size())
                    print(s_layer.weight.size())


                    #print(p_layer.weight.size()[0])
                    #print(expansion_factor * s_layer.weight.size()[0])
                    #print(p_layer.weight.size()[0] == expansion_factor * s_layer.weight.size()[0])
                    #print(p_layer.weight.size()[1] == s_layer.weight.size()[1])
                    #print(p_layer.bias.size()[0] == expansion_factor * s_layer.bias.size()[0])
                    #print(p_layer.bias.size()[1] == s_layer.bias.size[1])
                    #print(layer.bias.data.size())
                    pass

                elif layer_idx > 0 and layer_idx != len(prolonged_model.layers) - 1:
                    pass

                else:
                    pass




def test_lower_triangular_prolongation_weights(self, model):
    """
    Tests for weight and bias values using fixed seeds for Torch's random number generator
    on linear layers
    """
    for seed in range(10):

        # Set for deterministic results
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


        # Apply prolongation operator
        for expansion_factor in range(1):
            low_tri_op = prolongation.LowerTriangleProlongation(expansion_factor)
            prolonged_model = low_tri_op.apply(source_model = model)

            error_msg = "Prolonged copied weights are incorrect."

            # Check that prolonged weights are copied correctly
            for layer in prolonged_model.layers:
                print(layer.weight)
                print(layer.bias)
            """"
            for module_index in range(len(prolonged_model._module_layers)):
                mod_key = 'layer' + str(module_index)

                original_model_modlayers = model._module_layers[mod_key]
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

