""" MTNN/model.py
Defines the interface for creating extended torch.nn model
# TODO:
    * Logging: Change to literal string interpolation format (Python 3.6+)
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
import MTNN.torch_constants as torchconsts

# TODO: Logger INI file

logging.basicConfig(level = logging.DEBUG)

class Model(nn.Module):
    """
    Instantiates a neural network
    Args:
        config (str): The path to the YAML configuration file
        tensorboard (bool): Sets tensorboard visualization of the training
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
        tensorboard (bool): Sets data and control flow tracking through the optimizer to create a tensorboard visualization.
                See README.md for additional instructions on how to run Tensorboard.
        debug(bool): Sets additional logging to be collected when Model.fit is called.
    """
    # Tensorboard Writer
    WRITER = SummaryWriter('./runs/model/' + str(datetime.datetime.now()))  # Default is ./runs/model


    def __init__(self, config=None, tensorboard=None, debug=False):
        super(Model, self).__init__()
        self.config_file = config
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
        self.tensorboard = tensorboard
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
            layer_list = nn.ModuleList()
            this_layer = self._layer_config[n_layer]["layer"]
            prev_layer = self._layer_config[n_layer - 1]["layer"]
            layer_input = this_layer["neurons"]
            activation_type = this_layer["activation"]
            dropout = this_layer["dropout"]

            # Using Pytorch ModuleDict
            # Append hidden layer
            if n_layer == 0:  # First layer
                if this_layer["type"] == "linear":
                    layer_list.append(nn.Linear(self._input_size, layer_input))
            else:
                layer_list.append(nn.Linear(prev_layer["neurons"], layer_input))

            # Append activation layer
            try:
                layer_list.append(torchconsts.ACTIVATIONS[activation_type])
            except KeyError:
                print(str(activation_type) + " is not a valid torch.nn.activation function.")
                sys.exit(1)

            # TODO: Add Final softmax
            # TODO: Add dropout layers
            # TODO: Add support for a CNN

            # Update Layer dict
            layer_dict["layer" + str(n_layer)] = layer_list
            self._module_layers = layer_dict

            # Set model hyper-parameters
            self._hyperparameters = self.config_file["hyperparameters"]


        # Logging
        if self.debug:
            logging.debug(f"\n*****************************************************\
                        SETTING MODEL CONFIGURATION\
                        ********************************************************\
                        \nMODEL TYPE: {self._model_type}\
                        \nEXPECTED INPUT SIZE: {self._input_size}\
                        \nLAYERS: {self._module_layers}\
                        \nMODEL PARAMETERS:")
            for layer_idx in self._module_layers:
                logging.debug(f"\n\tLAYER: {layer_idx} \
                              \n\tWEIGHT: {self._module_layers[layer_idx][0].weight}\
                              \n\tWEIGHT GRADIENTS: {self._module_layers[layer_idx][0].weight.grad}\
                              \n\tBIAS: {self._module_layers[layer_idx][0].bias}\
                              \n\tBIAS GRADIENT: {self._module_layers[layer_idx][0].bias.grad}")

    def set_training_parameters(self, objective=None, optimizer=None):
        """
        Args:
            obj_fn <torch.nn.modules.loss>
            opt <torch.optim>

        Returns: Null

        """
        self._objective_fn = objective
        self._optimizer = optimizer

    def visualize(self, epoch=None, batch_num=None, log_interval=None, model_input=None, loss=None):
        self.WRITER.add_scalar('Train/Loss', loss, epoch)

        # Write to computation graph
        self.WRITER.add_graph(self, model_input)
        self.WRITER.flush()

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

        for module_indx in range(len(self._module_layers)):
            module_key = 'layer' + str(module_indx)

            layer = self._module_layers[module_key][0] # hard-coded. Assumes each module is one linear, one activation
            activation = self._module_layers[module_key][1]

            if not self.debug:
                model_input = layer(model_input)
                model_input = activation(model_input)

            if self.debug:
                logging.debug(f"FORWARD PASS LAYER #: {module_indx}\
                                \nINPUT: {model_input}")
                model_input = layer(model_input)
                model_input = activation(model_input)
                logging.debug(f"\n\tLAYER: {layer}\
                        \n\t\tWEIGHTS:\n\t\t {layer.weight}\
                        \n\t\tWEIGHT SHAPE:\n\t\t{layer.weight.size()}\
                        \n\t\tBIAS: \n\t\t{layer.bias}\
                        \n\t\tBIAS SHAPE: \n\t\t{layer.bias.size()}\
                        \n\t\tWEIGHTS GRADIENTS: \n\t\t {layer.weight.grad}\
                        \n\t\tBIAS GRADIENTS:\n\t\t {layer.bias.grad}")

            # TODO: Clear Logdir from previous runs
            # TODO: Disable asynchronous logging?

        # Visualize
        #TODO: FIX
        if self.tensorboard:
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

        self._train_count += 1


        return model_input

    def fit(self, dataloader, num_epochs, log_interval=1, checkpoint=False):
        """
        Train model.
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

        if self.debug:
            logging.debug("\n*********************************************************"
                          "STARTING TRAINING"
                          "***********************************************************")

        for epoch in range(1, num_epochs + 1):

            for batch_idx, (input_data, target_data) in enumerate(dataloader):

                # Reset weight/bias gradients
                self._optimizer.zero_grad()

                # TODO: Check and Reshape target size? Will be broadcasted if both tensors are broadcastable.

                if not self.debug:

                    # Forward pass
                    prediction = self.forward(input_data)
                    loss = self._objective_fn(prediction, target_data)
                    # Backward pass
                    loss.backward()
                    # Optimize pass - update weights with gradients
                    self._optimizer.step()

                # Train with logging
                else:
                    batch_total = len(dataloader)
                    logging.debug(f"\nEPOCH: {epoch}\
                                \nBATCH: {batch_idx}\
                                \nINPUT: {input_data}\
                                \nTARGET: {target_data}\
                                \n\tFORWARD PASS...\
                                \n\tBATCH [{batch_idx}/{batch_total}]")

                    prediction = self.forward(input_data)

                    logging.debug(f"PREDICTION: {prediction}")

                    loss = self._objective_fn(prediction, target_data)

                    logging.debug(f"LOSS: {loss}")
                    loss.backward(retain_graph = True)

                    self._optimizer.step()

                    logging.debug("UPDATED GRADIENTS")
                    # TODO: Add logging of weights after update?

                    logging.debug(f"FINISHED BATCH {batch_idx}")
                    logging.debug("*************************************************")

                # Print statistics.
                if not self.debug:
                    if batch_idx % (log_interval - 1) == 0 and batch_idx != 0:
                        print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(input_data), len(dataloader.dataset),
                                100. * batch_idx / len(dataloader), loss.item()))
                        train_losses.append(loss.item())
                        train_counter.append(
                            (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))

                else:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(input_data), len(dataloader.dataset),
                               100. * batch_idx / len(dataloader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))

                # Visualize.
                if self.tensorboard:
                    self.visualize(epoch=epoch, batch_num=batch_idx, model_input=input_data, loss=loss)

                # Save parameters.
                if checkpoint:
                    self.checkpoint()
                self._batch += 1

            self._epoch += 1
        self.WRITER.close()

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
            print("\n\tLayer: ", i,
                  "\n\tWeight: ", self._module_layers[i][0].weight,
                  "\n\tWeight Gradient", self._module_layers[i][0].weight.grad,
                  "\n\tBias: ", self._module_layers[i][0].bias,
                  "\n\tBias Gradient:", self._module_layers[i][0].bias.grad)

    # Accessor methods.
    def view_properties(self) -> list:
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


