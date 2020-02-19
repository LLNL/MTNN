""" MTNN/builder.py
Reads and Builds Model from YAML configuration files
# TODO: Add support to read JSON files
# TODO: Get tensorboard and debug settings from commandline
"""
# standard
import logging

# pytorch
import torch.nn as nn

# local source
import MTNN
import MTNN.config_reader as reader
import torch_constants as torchconsts


def build_model(confpath: str, visualize=False , debug=False):
    """
    Same functionality as MTNN.Model.set_config. Creates Layer_dict and sets model._module_layers
    Args:
        confpath: <str> Absolute file path of the model configuration file
        visualize: <bool> Used to set parameter collection for visualization
        debug: <bool> Used to set debugging logs to pring to console

    Returns:
        model: <MTNN.model>

    """
    conf = reader.YamlConfig(confpath)
    model = MTNN.Model(config=conf, visualize=visualize, debug=debug)

    # Set model attributes.
    model.set_model_type(conf.model_type)
    model.set_input_size(conf.input_size)
    model.set_layer_config(conf.layers)
    model.set_hyperparameters(conf.hyperparameters)
    model.set_objective(torchconsts.LOSS[conf.objective])

    # Process and set layers.
    layer_dict = nn.ModuleDict()
    layer_count = 0
    for n_layer in range(len(model._layer_config)):
        layer_module = nn.ModuleList()
        this_layer = model._layer_config[n_layer]["layer"]
        prev_layer = model._layer_config[n_layer - 1]["layer"]
        layer_input = this_layer["neurons"]
        activation_type = this_layer["activation"]
        dropout = this_layer["dropout"]

        # Using Pytorch ModuleDict
        # Append hidden layer
        if n_layer == 0:  # First layer
            if this_layer["type"] == "linear":
                layer_module.append(nn.Linear(model._input_size, layer_input))
        else:
            layer_module.append(nn.Linear(prev_layer["neurons"], layer_input))

        # Append activation layer
        try:
            layer_module.append(torchconsts.ACTIVATIONS[activation_type])
        except KeyError:
            print(KeyError,str(activation_type) + " is not a valid torch.nn.activation function.")

        # TODO: Add support for final layer softmax
        # TODO: Add support for dropout layers
        # TODO: Add Support for a CNN

        # Update Layer dict
        layer_dict["layer" + str(n_layer)] = layer_module

    model.set_num_layers(len(layer_dict))
    model._module_layers = layer_dict

    # Logging
    if debug:
        logging.debug(f"\n*****************************************************"
                    "SETTING MODEL CONFIGURATION"
                    "********************************************************")
        logging.debug(f"\nMODEL TYPE: {model._model_type}\
                        \nEXPECTED INPUT SIZE: {model._input_size}\
                        \nLAYERS: {model._module_layers}\
                        \nMODEL PARAMETERS:")
        for layer_idx in model._module_layers:
            logging.debug(f"\n\tLAYER: {layer_idx} \
                          \n\tWEIGHT: {model._module_layers[layer_idx][0].weight}\
                          \n\tWEIGHT GRADIENTS: {model._module_layers[layer_idx][0].weight.grad}\
                          \n\tBIAS: {model._module_layers[layer_idx][0].bias}\
                          \n\tBIAS GRADIENT: {model._module_layers[layer_idx][0].bias.grad}")

    return model

