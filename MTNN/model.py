# MTNN/model.py
"""
Defines the interface for creating extended torch.nn model
"""
import os
import sys
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

import logging



# Public API.
__all__ = ["Model"]


class Model(nn.Module):
    """
    Instantiates a neural network
    Attributes:
        ACTIVATION_TYPE (dict): available activation layers
    Args:
        config_file: a YAML configuration file
    """
    # Tensorboard Writer
    WRITER = SummaryWriter('./runs/model/' + str(datetime.datetime.now()))  # Default is ./runs

    OPTIMIZATIONS = {
        # TODO
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

    def __init__(self, tensorboard=None, debug=False):
        super(Model, self).__init__()
        self._config = None
        self._model_type = None
        self._input_size = None
        self._layer_num = 0
        self._layer_config = None
        self._objective_fn = None
        self._optimizer = None
        self._train_count = 0
        self._test_count = 0
        self._epoch = 0
        self._batch = 0
        self.tensorboard = tensorboard
        self.debug = debug

    # noinspection PyAttributeOutsideInit
    def set_config(self, yaml_filestream):
        self._config = yaml_filestream
        self.model_type = self._config["model-type"]
        self.input_size = self._config["input-size"]
        self.layer_num = self._config["number-of-layers"]
        self.layer_config = self._config["layers"]
        obj = self._config["objective"]
        opt = self._config["optimization"]

        # Process and set Layers.
        layer_dict = nn.ModuleDict()
        for n_layer in range(len(self.layer_config)):
            layerlist = nn.ModuleList()
            this_layer = self.layer_config[n_layer]["layer"]
            prev_layer = self.layer_config[n_layer-1]["layer"]
            layer_input = this_layer["neurons"]
            activation_type = this_layer["activation"]
            dropout = this_layer["dropout"]

            # Using ModuleDict
            if n_layer == 0:  # First layer
                if this_layer["type"] == "linear":
                    layerlist.append(nn.Linear(self.input_size, layer_input))
            else:
                layerlist.append(nn.Linear(prev_layer["neurons"], layer_input))

            # Add hidden activation layer
            try:
                layerlist.append(self.ACTIVATIONS[activation_type])
            except KeyError:
                print(str(activation_type) + " is not a valid torch.nn.activation function.")
                sys.exit(1)
            # TODO: Add Final softmax
            # TODO: Add dropout layers
            # TODO: Convolutional network
            
            layer_dict[str(n_layer)] = layerlist
            self.layers = layer_dict

            # Set hyperparameters.
            try:
                self._objective_fn = self.LOSS[obj]
            except KeyError:
                print(str(obj) + "is not a valid torch.nn.loss function")
                sys.exit(1)

            try:
                # TODO
                self._optimizer = self._config["optimization"]
            except KeyError:
                print(str(opt))

    def set_training_parameters(self, objective=None, optimizer=None):
        """
        Args:
            obj_fn <torch.nn.modules.loss>
            opt <torch.optim>

        Returns:

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

        Returns:
            model_input<torch.tensor>

        """
        raw_input = model_input
        for i, (layer, activation) in enumerate(self.layers.values()):
            # Reshape data
            # Input has to be a leaf variable to maintain gradients; no intermediate variables
            model_input = model_input.view(model_input.size(0), -1)
            model_input = layer(model_input)
            model_input = activation(model_input)
            if self.debug:
                logging.basicConfig(level = logging.DEBUG)
                logging.debug("\n\tINPUT: %s \n\tOUTPUT: %s",  raw_input, model_input)
                logging.debug ("\n\tWEIGHTS: %s \n\t WEIGHTSHAPE: %s \n\tBIAS: %s ", layer.weight , layer.weight.size(),layer.bias)
                logging.debug("\n\tWEIGHTS GRADIENTS: %s \n\tBIAS GRADIENTS: %s", layer.weight.grad, layer.bias.grad)
            # TODO: Clear Logdir from previous runs
            # TODO: Disable asynchronous logging?

        # Visualize
        if self.tensorboard:
            # Visualize weights
            for i, (layer, activation) in enumerate(self.layers.values()):
                for w_i, w in enumerate(layer.weight[0]):
                    self.WRITER.add_scalar('Train/Weights_' + str(w_i), layer.weight[0][w_i], self._epoch)
            # Visualize output
            model_output = model_input
            num_outputs = model_output.size()[1]
            for i in range(1, num_outputs):
                print(i)
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

        for epoch in range(1, num_epochs + 1):
            for batch_idx, (data, target) in enumerate(dataloader):

                # Zero out parameter gradients
                self._optimizer.zero_grad()

                if not self.debug:
                    # Forward + backward + optimize pass
                    prediction = self.forward(data)
                    loss = self._objective_fn(prediction, target)
                    loss.backward()

                    # Update weights with gradients
                    self._optimizer.step()
                else:
                    prediction = self.forward(data)
                    loss = self._objective_fn(prediction, target)
                    logging.basicConfig(level=logging.DEBUG)
                    logging.debug(" \nEPOCH: %i BATCH: %i TARGET:%s PREDICTION %s  LOSS: %s",
                                  epoch, batch_idx, target, prediction, loss)

                    loss.backward(retain_graph=True)
                    self._optimizer.step()
                    # TODO: Add logging of weights after update?

                # Print statistics.
                if batch_idx % (log_interval - 1 ) == 0 and batch_idx != 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))

                # Visualize.
                if self.tensorboard:
                    self.visualize(epoch=epoch, batch_num=batch_idx, model_input=data, loss=loss)

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

    def view_parameters(self): #
        print("Model Parameters are:")
        for i in self.layers:
            print("\n\tWeight: ",  self.layers[i][0].weight,
                  "\n\tWeight Gradient", self.layers[i][0].weight.grad,
                  "\n\tBias: ",  self.layers[i][0].bias,
                  "\n\tBias Gradient:", self.layers[i][0].bias.grad)



    def get_properties(self) -> list:
        model_properties = (self.model_type,
                            self.input_size,
                            self.layers,
                            self.output_activation_fn)
        return model_properties


