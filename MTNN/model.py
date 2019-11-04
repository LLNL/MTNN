# MTNN/model.py
"""
Defines the interface for creating extended torch.nn model
"""
import os
import sys
import datetime
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

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
    }

    def __init__(self, tensorboard=None):
        super(Model, self).__init__()
        self.tensorboard = tensorboard
        self._train_count = 0
        self._test_count = 0

    # noinspection PyAttributeOutsideInit
    def set_config(self, config_file):
        self._config = config_file
        self.model_type = self._config["model-type"]
        self.input_size = self._config["input-size"]
        self.layer_num = self._config["number-of-layers"]
        self.layer_config = self._config["layers"]
        self.output_activation_fn = self._config["output-activation-function"]

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
        self._train_count += 1
        return x

    def visualize(self, input, loss, epoch):
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()

        # Record training loss from each epoch into the writer
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_graph(self, input)
        writer.flush()

    def run_train(self, loss_fn, dataloader, opt, num_epochs=None, log_interval=None, checkpoint=False):
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

    def validate(self):
        # TODO: set train mode to false; model.eval()
        # TODO: Integrate with SGD_training?
        pass

    @property
    def get_properties(self) -> list:
        model_properties = (self.model_type,
                            self.input_size,
                            self.layers,
                            self.output_activation_fn)
        return model_properties


