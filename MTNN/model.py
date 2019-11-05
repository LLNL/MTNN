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
import numpy as np


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

    ACTIVATIONS = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax()
    }

    def __init__(self, tensorboard=None, debug=False):
        super(Model, self).__init__()
        self._train_count = 0
        self._test_count = 0
        self.tensorboard = tensorboard
        self.debug = debug

    # noinspection PyAttributeOutsideInit
    def set_config(self, config_file):
        self._config = config_file
        self.model_type = self._config["model-type"]
        self.input_size = self._config["input-size"]
        self.layer_num = self._config["number-of-layers"]
        self.layer_config = self._config["layers"]

        # Process and set Layers.
        layer_list = []
        layer_dict = nn.ModuleDict()
        for n_layer in range(len(self.layer_config)):
            layerlist = nn.ModuleList()
            this_layer = self.layer_config[n_layer]["layer"]
            prev_layer = self.layer_config[n_layer-1]["layer"]
            layer_input = this_layer["input"]
            activation_type = this_layer["activation"]
            dropout = this_layer["dropout"]

            # Using ModuleDict
            if n_layer == 0:  # First layer
                if this_layer["type"] == "linear":
                    layerlist.append(nn.Linear(self.input_size, layer_input))
            else:
                layerlist.append(nn.Linear(prev_layer["input"], layer_input))

            # Add hidden activation layer
            try:
                layerlist.append(self.ACTIVATIONS[activation_type])
            except KeyError:
                print(str(activation_type) + " is not a valid activation function.")
                sys.exit(1)
            # TODO: Add Final softmax
            # TODO: Add dropout layers
            # TODO: Convolutional network
            
            layer_dict[str(n_layer)] = layerlist
            self.layers = layer_dict



    def forward(self, x):

        for i, (layer, activation) in enumerate(self.layers.values()):
            # Reshape data
            x = x.view(x.size(0), -1)

            x = layer(x)
            x = activation(x)

        if self.debug:

            log_file_dir = os.getcwd() + '/debug/'
            if not os.path.exists(log_file_dir):
                os.mkdir(log_file_dir)
            log_file = open(log_file_dir + 'debug.txt', 'w')
            log_file.write(str(layer))
            np.savetxt(log_file, layer.weight.detach())
            log_file.close()
        self._train_count += 1

        return x

    def visualize(self, input, loss, epoch):
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()

        # Record training loss from each epoch into the writer
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_graph(self, input)
        writer.flush()

    def fit(self, loss_fn, dataloader, opt, num_epochs=None, log_interval=None, checkpoint=False):
        train_losses = []
        train_counter = []

        # Set children modules to training mode
        self.train()
        for epoch in range(1, num_epochs + 1):
            for batch_idx, (data, target) in enumerate(dataloader):
                # Zero out parameter gradients
                opt.zero_grad()

                # Forward + backward + optimize pass
                output = self.forward(data)
                loss = loss_fn(output, target)
                loss.backward()
                opt.step()

                # Print statistics
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item()))
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(dataloader.dataset)))

                # Visualize
                if self.tensorboard:
                    self.visualize(data, loss, epoch)

                # Save parameters
                if checkpoint:
                    self.checkpoint()

    def checkpoint(self):
        print('Saving model')
        checkpoint_dir = os.getcwd() + '/model/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')) + '.pth'
        with open(checkpoint_dir, 'wb') as f:
            torch.save(self.state_dict, f)

    def test(self, model, device, test_loader):
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

    def get_properties(self) -> list:
        model_properties = (self.model_type,
                            self.input_size,
                            self.layers,
                            self.output_activation_fn)
        return model_properties


