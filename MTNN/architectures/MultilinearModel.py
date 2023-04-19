import torch
from MTNN.architectures.architecture_interfaces import *

class MultilinearNet(BaseModel):
    def __init__(self, dim: list, activation, output_activation, weight_fill=None, bias_fill=None): # Check activationtype
        """
        Builds a fully connected network given a list of dimensions
        Args:
            dim: <list> List of dimensions [dim_in, hidden ,..., dim_out]
            activation: <torch.nn.Functional> Torch activation function
            output_activation: <torch.nn.Functional or lambda> Final function.
            weight_fill: <float> For debugging. Value to fill weights for each layer
            bias_fill: <float> For debugging. Value to fill bias for each layer
        """
        super().__init__()
        self.activation = activation
        self.output = output_activation
        self.dim = dim
       
        # Fill layers
        modules = nn.ModuleList()
        with torch.no_grad():
            for x in range(len(dim) - 1):
                layer = nn.Linear(dim[x], dim[x + 1])
                if weight_fill and bias_fill:
                    layer.weight.fill_(weight_fill)
                    layer.bias.fill_(bias_fill)
                modules.append(layer)

        self.layers = modules
        self.layers.to(self.device)

    def save_params(self, path):
        torch.save(self.layers, path)

    def load_params(self, path):
        self.layers = torch.load(path)

    def forward(self, x):
        # Flatten Input
        x = x.view(x.size(0), -1)

        for idx, layer in enumerate(self.layers):
            if idx != (len(self.layers) - 1):
                x = self.layers[idx](x)
                x = self.activation(x)

            elif layer == self.layers[-1]:
                x = self.layers[idx](x)
                x = self.output(x)

        return x

class MultilinearConverter(SecondOrderConverter):
    """MultiLinearConverter

    A SecondOrderConverter for multilinear, aka fully-connected, models.

    In this case, NN and MTNN formats are the same, so conversion is
    trivial.
    """
    def convert_network_format_to_MTNN(self, param_vector):
        pass

    def convert_MTNN_format_to_network(self, param_vector):
        pass


class CoarseMultilinearFactory(CoarseModelFactory):
    def build(self, fine_network, coarse_mapping):
        coarse_level_dims = [fine_network.layers[0].in_features] + coarse_mapping.num_coarse_channels + \
                            [fine_network.layers[-1].out_features]
        coarse_net = fine_network.__class__(coarse_level_dims,
                                              fine_network.activation,
                                              fine_network.output)
        coarse_net.set_device(fine_network.device)
        return coarse_net
