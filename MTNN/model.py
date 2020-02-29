""" MTNN/model.py
Defines the interface for creating extended torch.nn model
# TODO: Logger INI file
"""
# Public API.
__all__ = ["Model"]

# standard
import os
import sys
import datetime
import logging
import pprint

# third-party
import yaml

# pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

# local source
import MTNN.mtnn_defaults as mtnnconsts
import MTNN.torch_consts as torchconsts

# Debugging
#torch.manual_seed(1)

# Set-up logging
pp = pprint.PrettyPrinter(indent=4)
logging.basicConfig(filename=(mtnnconsts.EXPERIMENT_LOGS_FILENAME + ".log.txt"),
                    filemode='w',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)


class Model(nn.Module):
    """
    Instantiates a neural network
    Args:
        config (str): The path to the YAML configuration file
        visualize (bool): Sets tensorboard visualization of the training
        debug (bool): Sets logging during the the training.
            Logs are generated to the filepath specified in mtnn_defaults.py
    Attributes:
        config_file (str): File path to the YAML configuration file
        model_type (str): Specifies type of neural network architecture: Fully-Connected, CNN, etc.
        input_size (int): Specifies the expected input size to the neural network
        layer_config (list): List of layers specified in the YAML configuration file
        module_layers (nn.ModuleDict()): A dictionary of layers with the key being layer<num>
                and the value being a nn.ModuleList of layers. The inner layers can be
                "stacked" with layer objects from torch.nn (i.e. Linear, Relu)
        hyperparameters (dict): A dictionary of training parameters specified from the configuration file
        objective_fn (torch.nn.*Loss): A torch.nn Loss function. See https://pytorch.org/docs/stable/nn.html#loss-functions
        optimizer (torch.optim): An instantiated torch.optim.optimizer object specified by the YAML configuration file.
                Expected to be built with builder.build_optimizer. See https://pytorch.org/docs/stable/optim.html#
        visualize (bool): Sets data and control flow tracking through the optimizer to create a tensorboard visualization.
                See README.md for additional instructions on how to run Tensorboard.
        debug(bool): Sets additional logging to be collected when Model.fit is called.
    """
    # Tensorboard Writer
    WRITER = SummaryWriter('./runs/model/' + str(datetime.datetime.now()))  # Default is ./runs/model

    def __init__(self, config=None, visualize=None, debug=False):
        super(Model, self).__init__()
        # Public
        self.config_file = config # Todo: Remove
        self.num_layers = None

        # Protected
        self._model_type = None
        self._input_size = None
        self._layer_config = None
        self._module_layers = nn.ModuleDict()
        self._hyperparameters = None

        # Hyper-parameters
        self._objective_fn = None
        self._optimizer = None
        self._train_count = 0
        self._test_count = 0
        self._epoch = 0
        self._batch = 0

        # For debugging
        self.visualize = visualize
        self.debug = debug


    def set_config(self, config=mtnnconsts.DEFAULT_CONFIG):
        # TODO: Remove. Moved to Builder.py
        """
        Sets MTNN Model attributes from the YAML configuration file.
        Args:
            config: <str> File path for a YAML-formatted configuration file

        Returns:
            <None> MTNN.Model with set attributes.
        """

        if config:
            self.config_file = yaml.load(open(config, "r"), Loader=yaml.SafeLoader)

        self._model_type = self.config_file["model_type"]
        self._input_size = self.config_file["input_size"]
        self._layer_config = self.config_file["layers"]

        # Process and set Layers.
        layer_dict = nn.ModuleDict()
        for n_layer in range(len(self._layer_config)):
            module_list = nn.ModuleList()
            this_layer = self._layer_config[n_layer]["layer"]
            prev_layer = self._layer_config[n_layer - 1]["layer"]
            layer_output_size = this_layer["neurons"]
            activation_type = this_layer["activation"]
            dropout = this_layer["dropout"]

            # Using Pytorch ModuleDict
            # Append hidden layer
            if n_layer == 0:  # First layer
                if this_layer["type"] == "linear":
                    module_list.append(nn.Linear(self._input_size, layer_output_size))
            else:
                module_list.append(nn.Linear(prev_layer["neurons"], layer_output_size))

            # Append activation layer
            try:
                module_list.append(torchconsts.ACTIVATIONS[activation_type])
            except KeyError:
                print(str(activation_type) + " is not a valid torch.nn.activation function.")
                sys.exit(1)

            # TODO: Add Final softmax
            # TODO: Add dropout layers
            # TODO: Add support for a CNN



            # Update Layer dict
            layer_dict["layer" + str(n_layer)] = module_list
            self._module_layers = layer_dict

            # Set model hyper-parameters
            self._hyperparameters = self.config_file["hyperparameters"]

        # Logging
        if self.debug:
            logging.debug("\n*****************************************************"
                          "SETTING MODEL CONFIGURATION"
                          "********************************************************")
            logging.debug(f"\nMODEL TYPE: {self._model_type}\
                        \nEXPECTED INPUT SIZE: {self._input_size}\
                        \nLAYERS: {self._module_layers}\
                        \nMODEL PARAMETERS:")
            for layer_idx in self._module_layers:
                logging.debug(f"\n\tLAYER: {layer_idx} \
                              \n\tWEIGHT: {self._module_layers[layer_idx][0].weight}\
                              \n\tWEIGHT GRADIENTS: {self._module_layers[layer_idx][0].weight.grad}\
                              \n\n\tBIAS: {self._module_layers[layer_idx][0].bias}\
                              \n\tBIAS GRADIENT: {self._module_layers[layer_idx][0].bias.grad}")

    def forward(self, model_input):
        """
        Forward pass on the model to compute gradients.
        Args:
            model_input <torch.tensor>

        Returns: None

        """

        # Reshape input data
        # Note: Input has to be a leaf variable to maintain gradients; no intermediate variables
        model_input = model_input.view(model_input.size(0), -1)

        # Without logging
        if not self.debug:
            for module_indx in range(len(self._module_layers)):
                module_key = 'layer' + str(module_indx)

                layer = self._module_layers[module_key][0] # hard-coded. Assumes each module is one linear, one activation
                activation = self._module_layers[module_key][1]

                model_input = layer(model_input)
                model_input = activation(model_input)

        # With logging
        if self.debug:
            for module_indx in range(len(self._module_layers)):
                module_key = 'layer' + str(module_indx)

                for layer in self._module_layers[module_key]:
                    logging.debug(f"\tLayer: {layer}")
                    logging.debug(f"\t\tINPUT: {model_input}")
                    model_input = layer(model_input)
                    logging.debug(f"\t\tOUTPUT: {model_input}")

                    # TODO:FIX. Non-leaf/intermediate variables gradients can only be accessed via hooks.
                    if hasattr(layer, "weight"):
                        logging.debug(f"\n\t\tLAYER: {layer}\
                                               \n\t\t\tWEIGHTS:\n\t\t\t\t\t{layer.weight.data} {layer.weight.requires_grad}\
                                               \n\t\t\tWEIGHT SHAPE:\n\t\t\t\t{layer.weight.size()}\
                                               \n\t\t\tBIAS:\n\t\t\t\t{layer.bias.data} {layer.bias.requires_grad}\
                                               \n\t\t\tBIAS SHAPE: \n\t\t\t\t{layer.bias.size()}\
                                               \n\t\t\tWEIGHTS GRADIENTS:\n\t\t\t\t{layer.weight.grad}\
                                               \n\t\t\tBIAS GRADIENTS:\n\t\t\t\t{layer.bias.grad}")


        # Visualize
        if self.visualize:
            """
            # For Tensorboard Visualization
            # Visualize weights
            for i, (layer, activation) in enumerate(self._module_layers.values()):
                for w_i, w in enumerate(layer.weight[0]):
                    self.WRITER.add_scalar('Train/Weights_' + str(w_i), layer.weight[0][w_i], self._epoch)
            # Visualize output
            model_output = model_input
            num_outputs = model_output.size()[1]
            # TODO: Tracerwarning: Converting a tensor to  Python index might cause the trace to be incorrect.
            # This value will be treated as a constant in the future and not be recorded as part of the data flow graph.
            for i in range(1, num_outputs):
                self.WRITER.add_scalar('Train/Output_' + str(i), model_output.data[0][i], self._epoch)
            """
            #TODO: Add forward hooks
            pass

        self._train_count += 1

        return model_input

    def fit(self, dataloader, num_epochs, log_interval=1, checkpoint=False):
        """
        Train model given dataloader. Calls Forward pass.
        Args:
            dataloader <torch.utils.data.DataLoader>
            num_epochs <int>
            log_interval <int>
            checkpoint <bool>

        Returns:

        """
        train_losses = []
        train_counter = []

        # Set children modules to training mode
        self.train()

        # Add hooks to collect plotting data
        if self.visualize:
            print("\nVISUALIZTION")
            #print(self._modules.items())

        # Train with Logging.
        if self.debug:
            logging.debug("\n*********************************************************"
                          "STARTING TRAINING"
                          "***********************************************************")
            for epoch in range(1, num_epochs + 1):
                for batch_idx, (input_data, target_data) in enumerate(dataloader):
                    # Reset weight/bias gradients
                    self._optimizer.zero_grad()

                    batch_total = len(dataloader)
                    logging.debug(f"\nEPOCH: {epoch}\
                                                   \nBATCH: {batch_idx}\
                                                   \nINPUT: {input_data}\
                                                   \nTARGET: {target_data}\
                                                   \n\tFORWARD PASS...\
                                                   \n\tBATCH [{batch_idx}/{batch_total}]")

                    prediction = self.forward(input_data)
                    logging.debug(f"\nPREDICTION: {prediction}")

                    # Back propagation
                    loss = self._objective_fn(prediction, target_data)
                    loss.backward(retain_graph = True) # Compute gradients wrt to parameters in loss
                    logging.debug(f"\nLOSS: {loss}")

                    # Gradient Descent

                    logging.debug(f"\nOPTIMIZER STATE # of PARAMS: {len(self._optimizer.state)}")

                    for k, v in enumerate(self._optimizer.state):
                        logging.debug(f"\t\tKEY: {k} VALUE: {v}")

                    logging.debug("\n\tOPTIMIZER PARAMETERS)")
                    for p in self._optimizer.param_groups[0]["params"]:
                        logging.debug(f"\t {p.data} \trequires_grad={p.requires_grad}")

                    self._optimizer.step()# Update parameters

                    logging.debug("\nUPDATED WEIGHTS AND BIASES..")
                    for module_indx in range(len(self._module_layers)):
                        module_key = 'layer' + str(module_indx)
                        for layer in self._module_layers[module_key]:
                            # TODO:FIX. Non-leaf/intermediate variables gradients can only be accessed via hooks.
                            if hasattr(layer, "weight"):
                                logging.debug(f"\n\t\tLAYER: {layer}\
                                                                  \n\t\t\tWEIGHTS:\n\t\t\t\t\t{layer.weight.data} {layer.weight.requires_grad}\
                                                                  \n\t\t\tWEIGHT SHAPE:\n\t\t\t\t{layer.weight.size()}\
                                                                  \n\t\t\tBIAS:\n\t\t\t\t{layer.bias.data} {layer.bias.requires_grad}\
                                                                  \n\t\t\tBIAS SHAPE: \n\t\t\t\t{layer.bias.size()}\
                                                                  \n\t\t\tWEIGHTS GRADIENTS:\n\t\t\t\t{layer.weight.grad}\
                                                                  \n\t\t\tBIAS GRADIENTS:\n\t\t\t\t{layer.bias.grad}")

                    # TODO: Add logging of weights after update?
                    logging.debug("")

                    logging.debug(f"FINISHED BATCH {batch_idx}")
                    logging.debug("*************************************************")
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_data), len(dataloader.dataset),
                               100. * batch_idx / len(dataloader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))
                    # Save parameters.
                    if checkpoint:
                        self.checkpoint()
                    self._batch += 1
                self._epoch += 1
            self.WRITER.close()

        # Train without Logging
        if not self.debug:
            for epoch in range(1, num_epochs + 1):
                for batch_idx, (input_data, target_data) in enumerate(dataloader):

                    # Reset weight/bias gradients
                    self._optimizer.zero_grad()

                    # TODO: Check and Reshape target size? Will be broadcasted if both tensors are broadcastable.

                    # Forward pass
                    prediction = self.forward(input_data)
                    loss = self._objective_fn(prediction, target_data)
                    # Backward pass
                    loss.backward()
                    # Optimize pass - update weights with gradients
                    self._optimizer.step()

                    # Print statistics.
                    if batch_idx % (log_interval - 1) == 0 and batch_idx != 0:
                        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(input_data), len(dataloader.dataset),
                                100. * batch_idx / len(dataloader), loss.item()))
                        train_losses.append(loss.item())
                        train_counter.append(
                            (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))

                    # Save parameters.
                    if checkpoint:
                        self.checkpoint()
                    self._batch += 1
                self._epoch += 1

    def visualize(self, epoch=None, batch_num=None, log_interval=None, model_input=None, loss=None):
        # TODO: Remove. Dead code for Tensorboard
        self.WRITER.add_scalar('Train/Loss', loss, epoch)

        # Write to computation graph
        self.WRITER.add_graph(self, model_input)
        self.WRITER.flush()

    def checkpoint(self):
        print('Saving model')
        checkpoint_dir = os.getcwd() + '/model/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')) + '.pth'
        with open(checkpoint_dir, 'wb') as f:
            torch.save(self.state_dict, f)

    def validate(self, model, device, test_loader):
        # TODO: Integrate with SGD_training
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction = 'sum').item()  # sum up batch loss
                    pred = output.argmax(dim = 1, keepdim = True)  # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    def print_parameters(self):
        """
        Prints out weights and biases
        Returns: None
        """
        print("\n MODEL PARAMETERS:")
        for i in self._module_layers:
            print("\n\tLAYER: ", i,
                  "\n\tWEIGHT:\n\t", self._module_layers[i][0].weight.data,
                  "\n\tWEIGHT GRADIENT", self._module_layers[i][0].weight.grad,
                  "\n\n\tBIAS:\n\t", self._module_layers[i][0].bias.data,
                  "\n\tBIAS GRADIENT:", self._module_layers[i][0].bias.grad)

    # Accessor methods.
    def print_properties(self) -> list:
        """
        Returns model configuration
        Returns: None
        """
        model_properties = f'\n MODEL TYPE: {self._model_type}' \
                           f'\n INPUT SIZE: {self._input_size}' \
                           f'\n HYPER-PARAMETERS: {self._hyperparameters}' \
                           f'\n MODULE LAYERS: {self._module_layers}'\
                           f'\n OBJECTIVE FUNCTION: {self._objective_fn}'\
                           f'\n OPTIMIZATION: {self._optimizer}'
        print(model_properties)
        return



    # Mutator Methods.
    def set_debug(self, debug: bool):
        self.debug = debug

    def set_model_type(self, model_type: str):
        self._model_type = model_type

    def set_input_size(self, input_size: int):
        self._input_size = input_size

    def set_layer_config(self, layer_config: list):
        self._layer_config = layer_config

    def set_num_layers(self, num_layers: int):
        self.num_layers = num_layers

    def set_module_layers(self, module_dict: '<class nn.ModuleDict()>'):
        self._module_layers = module_dict

    def set_hyperparameters(self, hyperparameters: list):
        self._hyperparameters = hyperparameters

    def set_objective(self, objective_fn: '<class torch.nn.modules.loss>'):
        # Check type
        if issubclass(objective_fn.__class__, torch.nn.modules.loss._Loss().__class__):
            self._objective_fn = objective_fn

    def set_optimizer(self, optimizer: '<class torch.optim>'):
        # TODO: Check type
        #if issubclass(optimizer.__class__, torch.optim.Optimizer().__class__):
        self._optimizer = optimizer

    def set_training_parameters(self, objective=None, optimizer=None):
        """
        Args:
            obj_fn <torch.nn.modules.loss>
            opt <torch.optim>

        Returns: Null

        """
        self._objective_fn = objective
        self._optimizer = optimizer


