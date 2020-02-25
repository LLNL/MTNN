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
        module_list = nn.ModuleList()
        this_layer = model._layer_config[n_layer]["layer"]
        prev_layer = model._layer_config[n_layer - 1]["layer"]
        layer_output_size = this_layer["neurons"]
        activation_type = this_layer["activation"]
        dropout = this_layer["dropout"]

        # Using Pytorch ModuleDict
        # Append hidden layer
        if n_layer == 0:  # First layer
            if this_layer["type"] == "linear":
                first_linear_layer = nn.Linear(model._input_size, layer_output_size)
                logging.debug(f"Layer:  {first_linear_layer.weight} {first_linear_layer.bias}")
                module_list.append(first_linear_layer)
        else:
            linear_layer = nn.Linear(prev_layer["neurons"], layer_output_size)
            logging.debug(f"Layer: {linear_layer.weight} {linear_layer.bias}")
            module_list.append(linear_layer)

        # Append activation layer
        try:
            module_list.append(torchconsts.ACTIVATIONS[activation_type])
        except KeyError:
            print(KeyError,str(activation_type) + " is not a valid torch.nn.activation function.")

        # TODO: Add support for final layer softmax
        # TODO: Add support for dropout layers
        # TODO: Add Support for a CNN

        # Add to Layer dict 
        layer_dict["layer" + str(n_layer)] = module_list

    # Update Model attributes
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

