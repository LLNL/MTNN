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
import tests_var


class Model(nn.Module):
    """
    Instantiates a neural network
    Attributes:
        LOSS <dict>: available PyTorch loss functions
        OPTIMIZATION <dict>: available PyTorch optimization functions
        ACTIVATION_TYPE <dict>: available PyTorch activation layers
    """

    # TODO: Refactor. Move global variables to __init__.py or config.ini

    # Tensorboard Writer
    WRITER = SummaryWriter('./runs/model/' + str(datetime.datetime.now()))  # Default is ./runs/model

    # TODO: make case-insensitive
    OPTIMIZATIONS = {
        # TODO: fill-in but instantiate?
    }

    LOSS = {
        "crossentropy": nn.CrossEntropyLoss(),
        "mseloss": nn.MSELoss()
    }

    ACTIVATIONS = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax()
    }


    def __init__(self, config=None, tensorboard=None, debug=False):
        super(Model, self).__init__()
        self.config_file = config
        self._model_type = None
        self._input_size = None
        self._num_layers = 0
        self._layer_config = None
        self._module_layers = nn.ModuleDict()

        # Hyper-parameterS
        self._objective_fn = None
        self._optimizer = None
        self._train_count = 0
        self._test_count = 0
        self._epoch = 0
        self._batch = 0

        # For debugging
        self.tensorboard = tensorboard
        self.debug = debug
        if self.debug:
            logging.basicConfig(level = logging.DEBUG)


    def set_config(self, config=tests_var.DEFAULT_CONFIG):
        """
        Sets MTNN Model attributes from the YAML configuration file.
        Args:
            config: <str> File path for a YAML-formatted configuration file

        Returns:
            <None> MTNN.Model with set attributes.
        """

        if config:
            self.config_file = yaml.load(open(config, "r"), Loader=yaml.SafeLoader)

        #TODO: validate keys
        self._model_type = self.config_file["model_type"]
        self._input_size = self.config_file["input_size"]
        # TODO: Refactor: Remove num_layers attribute. Get from len(config[layers])
        self._num_layers = self.config_file["number_of_layers"]
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

            # Using ModuleDict
            # Append hidden layer
            if n_layer == 0:  # First layer
                if this_layer["type"] == "linear":
                    layer_list.append(nn.Linear(self._input_size, layer_input))
            else:
                layer_list.append(nn.Linear(prev_layer["neurons"], layer_input))

            # Append activation layer
            try:
                layer_list.append(self.ACTIVATIONS[activation_type])
            except KeyError:
                print(str(activation_type) + " is not a valid torch.nn.activation function.")
                sys.exit(1)
            # TODO: Add Final softmax
            # TODO: Add dropout layers
            # TODO: Convolutional network

            layer_dict["layer" + str(n_layer)] = layer_list
            self._module_layers = layer_dict

            obj = self.config_file["objective"]
            opt = self.config_file["optimization"]

            # Objective function
            try:
                self._objective_fn = self.LOSS[obj]
            except KeyError:
                print(str(obj) + "is not a valid torch.nn.loss function")
                sys.exit(1)

            # Optimizer
            try:
                # TODO: Is this even a good idea?
                self._optimizer = self.config_file["optimization"]
            except KeyError:
                print(str(opt))

        # Logging

        if self.debug:
            logging.debug("\n*****************************************************"
                          "SETTING MODEL CONFIGURATION"
                         "********************************************************")
            logging.debug(f"\nMODEL TYPE: {self._model_type}")
            logging.debug(f"\nEXPECTED INPUT SIZE: {self._input_size}")
            logging.debug(f"\nLAYERS: {self._module_layers}")
            logging.debug("\nMODEL PARAMETERS:")
            for layer_idx in self._module_layers:
                logging.debug(f"\n\tLAYER: {layer_idx}")
                logging.debug(f"\n\tWEIGHT: {self._module_layers[layer_idx][0].weight}")
                logging.debug(f"\n\tWEIGHT GRADIENTS: {self._module_layers[layer_idx][0].weight.grad}")
                logging.debug(f"\n\tBIAS: {self._module_layers[layer_idx][0].bias}")
                logging.debug(f"\n\tBIAS GRADIENT: {self._module_layers[layer_idx][0].bias.grad}")



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
                logging.debug(f"FORWARD PASS LAYER #: {module_indx} ")
                logging.debug(f"\nINPUT: {model_input}")
                model_input = layer(model_input)
                model_input = activation(model_input)
                logging.debug(f"\n\tLAYER: {layer}")
                logging.debug(f"\n\t\tWEIGHTS:\n\t\t {layer.weight}")
                logging.debug(f"\n\t\tWEIGHT SHAPE:\n\t\t{layer.weight.size()}") #Tracerwarning
                logging.debug(f"\n\t\tBIAS: \n\t\t{layer.bias}")
                logging.debug(f"\n\t\tBIAS SHAPE: \n\t\t{layer.bias.size()}") #Tracerwarning
                logging.debug(f"\n\t\tWEIGHTS GRADIENTS: \n\t\t {layer.weight.grad}")
                logging.debug(f"\n\t\tBIAS GRADIENTS:\n\t\t {layer.bias.grad}")

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
                    logging.debug(f"\nEPOCH: {epoch} BATCH: {batch_idx}")
                    logging.debug(f"INPUT: {input_data}")
                    logging.debug(f"TARGET: {target_data}")

                    logging.debug("FORWARD PASS...")
                    logging.debug(f"BATCH [{batch_idx}/{batch_total}]")
                    prediction = self.forward(input_data)

                    logging.debug(f"PREDICTION: {prediction}")

                    logging.debug("\nCALCULATING LOSS...")
                    loss = self._objective_fn(prediction, target_data)

                    logging.debug(f"LOSS: {loss}")
                    loss.backward(retain_graph = True)

                    logging.debug("UPDATING GRADIENTS...")
                    self._optimizer.step()

                    logging.debug("UPDATED GRADIENTS")
                    # TODO: Add logging of weights after update?

                logging.debug(f"FINISHED BATCH {batch_idx}")
                logging.debug("*************************************************")

                # Print statistics.
                if batch_idx % (log_interval - 1) == 0 and batch_idx != 0:
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

    def get_properties(self) -> list:
        """
        Returns model configuration
        Returns: None
        """
        model_properties = (self._model_type,
                            self._input_size,
                            self._module_layers)
        return model_properties

    # Mutator Methods.
    def set_debug(self, debug: bool):
        self.debug = debug


