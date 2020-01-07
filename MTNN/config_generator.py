"""Script to generate permutations of fully-connected neural network yaml files"""
#!/usr/bin/env python

# System package
import itertools
import yaml

# Local package
from MTNN import CONFIG_DIR, CONFIG_PARAMETERS


class Layer:
    """Base Layer yaml template class
    Used to set layer parameters to append to the model template.
    """
    def __init__(self, layertype: str, numneurons: int, activationtype: str, dropout: bool):
        self.type = layertype
        self.neurons = numneurons
        self.activation = activationtype
        self.dropout = dropout

    # Mutator methods.
    # are these needed if __setattr__ already built in?
    def set_type(self, model_type: str):
        self.type = model_type

    def set_activation(self, activation: str):
        self.activation = activation

    def set_neurons(self, num_neurons):
        self.neurons = num_neurons

    def set_dropout(self, dropout: str): # Check pytorch type
        self.dropout = dropout

    # Accessor method.
    def get_attributes(self):
        print("Type: ", self.type)
        print("Neurons: ", self.neurons)
        print("Activation: ", self.activation)
        print("Dropout: ", self.dropout)

    def write_as_yaml(self) -> str:
        """ Writes Layer object attributes as a string in this format:
             {'layer':
                 {'type': "linear",
                  'neurons': 0,
                  'activation': "relu",
                  'dropout': None
                  }
             }
        """
        layer_attr = vars(self)
        layer_dict = {'layer': layer_attr}
        return layer_dict


class Model:
    """ Base model yaml template class

    Used to set the model parameters.
    """
    def __init__(self, modeltype: str, inputsize: int, numlayers: int, layerlist: list):
        """
        Model yaml template
        Args:
            modeltype: <str> Use "fully-connected" for a fully connected model
            inputsize: <int> input size
            numlayers: <int> Number of layers
            layerlist: <list<dict>> List of dict to easily dump to string
        """
        self.model_type = modeltype
        self.input_size = inputsize
        self.num_layers = numlayers
        self.layers = layerlist

    # Mutator methods.
    def set_model_type(self, model: str):
        self.model_type = model

    def set_input(self, input_size: int):
        self.input_size = input_size

    def set_num_layers(self, num_layers: int):
        self.num_layers = num_layers

    def append_layer(self, layer: Layer):
        self.layers.append(layer)

    def write_as_yaml(self) -> str:
        """Writes Model object attributes as string in this format:
            {'model-type': "",
              'input-size': 0,
              'number-of-layers': 0,
              'layers': []
            }
        """
        model_dict = vars(self)
        return model_dict


########################################################
# Helper functions
########################################################
class NoAliasDumper(yaml.SafeDumper):
    """ Used to call yaml.dump without anchors and aliases errors
    """
    def ignore_aliases(self, _data):
        return True


def get_layer_data(layer_list: list):
    """ Used to process Layer data into string.
     Pre-processes data to be used in Model object.
     Gets list of Layer objects and returns a list of dicts.
    Args:
        layerlist <list<Layers>>
    Returns:
        layerData <list of dicts>
    """
    layer_data = []
    for layer in layer_list:
        layer_data.append(layer.write_as_yaml())
    return layer_data


def write_to_file(filename, modeldata: str):
    """
    Writes model data (string) to file
    Args:
        filename <filestream>: yaml file to be written to
        modeldata <str>: model yaml template

    Returns:

    """
    with open(filename, 'w') as file:
        yaml.dump(modeldata, file, encoding='utf8', allow_unicode=True, Dumper=NoAliasDumper)


def gen_config(parameters: dict, product: bool):
    """
    Generate fully connected network YAML configuration files
     with option to permute of # of neurons per layer.
    Args:
        parameters <dict>: network parameters bounds
        product <bool>: cartesian products of # layers * neuron range

    Returns:
        Writes a YAML file to config_directory. An example file name of "test123.yaml"
        is a network with 1 layer, 2 neurons and an input size of 3 values.
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

    # Make fully connected model configs with uniform neuron layers.
    for num_layer in range(layer_min, layer_max + 1):
        for num_neuron in range(neuron_min, neuron_max + 1):
            for model_input in range(input_min, input_max + 1):

                file_name = CONFIG_DIR + "/test" + str(num_layer) + str(num_neuron) + \
                            str(model_input) + ".yaml"

                # Create the layer.
                a_layer = Layer(layertype="linear",
                                numneurons=num_neuron,
                                activationtype="relu",
                                dropout=False)

                # Append layers.
                list_of_layers = []
                for p in range(num_layer):
                    list_of_layers.append(a_layer)

                layer_data = get_layer_data(list_of_layers)

                # Create/overwrite Model Instance.
                a_model = Model(modeltype="fully-connected",
                                inputsize=model_input,
                                numlayers=num_layer,
                                layerlist=layer_data)

                # Format data as string
                data = a_model.write_as_yaml()

                # Write to file.
                write_to_file(file_name, data)

    if product:
        """ Generate layers with varying layer depth in a fully-connected network.
        Ex. test322p#.yaml will have the following suffix denoting the of number neurons per layer:
        111, 121, 122, etc.

        File test322p222.yaml has 3 layers with 2 neuron each
        File test322p121.yaml has 3 layers, one with 1 neuron, 2 neuron, 1 neuron
        """
        for num_layer in range(layer_min, layer_max + 1):
            for num_neuron in range(neuron_min, neuron_max + 1):

                if num_layer > 1 and num_neuron > 1:
                    print("Product: layer:", num_layer, "neurons:", CONFIG_PARAMETERS["neurons"])

                    # Get number of permutations.
                    neuron_range = range(1, num_neuron + 1)
                    product = itertools.product(neuron_range, repeat=num_layer)  # cartesian product

                    # Generate the permuted layers.
                    for a_prod in list(product):
                        list_of_layers = []

                        for p in a_prod:
                            # Set layer neurons and append.
                            a_layer = Layer(layertype="linear",
                                            numneurons=p,
                                            activationtype="relu",
                                            dropout=False)
                            list_of_layers.append(a_layer)

                        layer_data = get_layer_data(list_of_layers)

                        # Create the model.
                        for model_input in range(input_min, input_max + 1):

                            a_model = Model(modeltype="fully-connected",
                                            inputsize=model_input,
                                            numlayers=num_layer,
                                            layerlist=layer_data)

                            # Format model data to string.
                            data = a_model.write_as_yaml()

                            # Write to file.
                            file_name = CONFIG_DIR + "/test" + str(num_layer) + str(num_neuron) \
                                        + str(model_input) + "p" \
                                        + str(list(a_prod)).replace(", ", "").strip('[]') + ".yaml"
                            print(file_name)
                            write_to_file(file_name, data)


def main():
    """ Generates variations of fully connected model yaml configuration files.
     Edit configuration parameters in CONFIG_PARAMETERS variable.
     Configuration files are written to path stored in CONFIG_DIR variable.
    """
    # Write model yaml config files out to CONFIG_DIR/config
    gen_config(CONFIG_PARAMETERS, product=True)


if __name__ == "__main__":
    main()