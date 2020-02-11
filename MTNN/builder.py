""" MTNN/builder.py
Reads and Builds Model from YAML configuration files
# TODO: Add support to read JSON files
# TODO: Get tensorboard and debug settings from commandline
"""
# pytorch
import torch.nn as nn
import torch.optim as optim

# local source
import MTNN
import MTNN.config_reader as reader
import torch_builtins as torchconsts


def build_model(confpath: str):
    """
    Same functionality as MTNN.Model.set_config. Creates Layer_dict and sets model._module_layers
    Args:
        confpath:

    Returns:
        model: <MTNN.model>

    """
    conf = reader.YamlConfig(confpath)
    model = MTNN.Model(config = conf, tensorboard = False, debug = True)

    # Set model parameters.
    model.set_model_type(conf.model_type)
    model.set_input_size(conf.input_size)
    model.set_layer_config(conf.layers)
    model.set_hyperparameters(conf.hyperparameters)
    model.set_objective(conf.objective)

    # Process and set layers.
    layer_dict = nn.ModuleDict()
    for n_layer in range(len(model._layer_config)):
        layer_list = nn.ModuleList()
        this_layer = model._layer_config[n_layer]["layer"]
        prev_layer = model._layer_config[n_layer - 1]["layer"]
        layer_input = this_layer["neurons"]
        activation_type = this_layer["activation"]
        dropout = this_layer["dropout"]

        # Using Pytorch ModuleDict
        # Append hidden layer
        if n_layer == 0:  # First layer
            if this_layer["type"] == "linear":
                layer_list.append(nn.Linear(model._input_size, layer_input))
        else:
            layer_list.append(nn.Linear(prev_layer["neurons"], layer_input))

        # Append activation layer
        try:
            layer_list.append(torchconsts.ACTIVATIONS[activation_type])
        except KeyError:
            print(KeyError,str(activation_type) + " is not a valid torch.nn.activation function.")

        # TODO: Add support for final layer softmax
        # TODO: Add support for dropout layers
        # TODO: Add Support for a CNN

        # Update Layer dict
        layer_dict["layer" + str(n_layer)] = layer_list
        model._module_layers = layer_dict
    return model

