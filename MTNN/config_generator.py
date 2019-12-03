"""
Script to generate permutations of fully-connected neural network yaml files
"""
#!/usr/bin/env python
import os
from itertools import permutations
import yaml


# Set file paths.
path = os.getcwd()
config_directory = path + "/config/"

if not os.path.exists(config_directory): #and os.path.isdir(config_directory):
    os.mkdir(config_directory)
    print(config_directory)


# Dictionary templates for YAML files
model_template = {'model-type': "fully-connected",
                   'input-size': 0,
                   'number-of-layers': 0,
                   'layers': []
                   }

layer_template = {'layer':
                       {'type': "linear",
                        'neurons': 0,
                        'activation': "relu",
                        'dropout': None
                        }
                   }


class NoAliasDumper(yaml.SafeDumper):
    # Used for yaml.dump without anchors and aliases
    def ignore_aliases(self, _data):
        return True


def gen_config(parameters:dict, permute:bool):
    """
    Generate different fully conencted network combinations of layers, inputs, and neurons.
    Args:
        parameters: <dict>

    Returns:
        Writes a YAML file to config_directory. An example file name of "test123.yaml"
        is a network with 1 layer, 2 neurons and an input size of 3 floats.

    """
    # Layers
    layer_min = parameters["layers"][0]
    layer_max = parameters["layers"][1]

    # Neurons/nodes
    neuron_min = parameters["neurons"][0]
    neuron_max = parameters["neurons"][1]

    # Input size
    input_min = parameters["input"][0]
    input_max = parameters["input"][1]

    for l in range(layer_min, layer_max + 1):
        for n in range(neuron_min, neuron_max + 1):
            for i in range(input_min, input_max + 1):
                file_name = "test" + str(l) + str(n) + str(i)

                model_template["number-of-layers"] = l
                model_template["input-size"] = i

                # Generate the layers.
                layer_template["layer"]["neurons"] = n
                list_of_layers = [layer_template] * l

                # Append.
                model_template["layers"].append(list_of_layers)

                # Write to file.
                with open(config_directory + file_name + ".yml", 'w') as file:
                    yaml.dump(model_template, file, encoding='utf8', allow_unicode=True, Dumper=NoAliasDumper)

                # Clear layers list.
                model_template["layers"].clear()
                print(model_template["layers"])

    if permute:
        # TODO: permutation of neurons between layers
        pass

# Edit this.
config_parameters = {
    "layers": (1, 3),
    "input": (1, 3),
    "neurons": (1, 3)
}

gen_config(config_parameters, True)


