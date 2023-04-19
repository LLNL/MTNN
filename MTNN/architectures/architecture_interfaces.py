from abc import abstractmethod
import torch.nn as nn
from MTNN.utils import logger, deviceloader
from MTNN.utils.datatypes import CoarseMapping

class BaseModel(nn.Module):
    """
    A neural network model in PyTorch
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.device = deviceloader.get_device(verbose=False)
        self.log = logger.get_MTNN_logger()

    def set_device(self, device):
        self.layers.to(device)

    def log_model(self): ### DELETE
        self.log.warning("Initial (Level 0) Model".center(100, '='))
        self.log.warning("Model type: {}".format(self.__class__.__name__))
        for layer_ind, layer in enumerate(self.layers):
            self.log.warning("Layer {} \t Layer type {} \t Weight {} \t Bias {}".format(
                layer_ind, layer.__class__.__name__, layer.weight.size(), layer.bias.size()))

    @abstractmethod
    def forward(self, x):
        """Apply this neural network to some input.

        @param x The torch.Tensor containing the input vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def save_params(self, path):
        """Save the parameters of this neural network to the file pointed to by `path`."""
        raise NotImplementedError

    @abstractmethod
    def load_params(self, path):
        """Load the parameters for this neural network from the file pointed to by `path`."""
        raise NotImplementedError

class CoarseModelFactory:
    """Constructs coarse neural networks based on fine networks and coarse
    mapping information.

    """

    @abstractmethod
    def build(self, fine_network, coarse_mapping):
        """Build an uninitialized coarse model.

        @param fine_network The neural network from the fine level.

        @param coarse_mapping a CoarseMapping object which describes
        how to map fine neurons to coarse neurons.

        """
        raise NotImplementedError


class SecondOrderConverter:
    """Converts parameter libraries between format.

    We currently support "second order restrictions," in which we
    apply restriction and prolongation via matrix multiplication on
    either side of a tensor. That is, if a parameter tensor W has
    dimensions $(d_1, d_2, ..., d_k, m, n)$, then we restrict via
    $R_op @ W @ P_op$, which, for each choice of indices $(i_1, ...,
    i_k)$ over the first $k$ dimensions, performs matrix
    multiplicaiton over the last two dimensions. We think of this as
    "second order" because the restriction operation is quadratic in
    $(R_op, P_op)$.

    A ParamVector contains lists of weight and bias tensors. The
    format of these tensors as required by the neural networks is
    different than the format required by PyTorch for its matrix
    multiplication broadcasting semantics, which we use during
    restriction and prolongation. This class converts between these
    two formats.

    """
    
    @abstractmethod
    def convert_network_format_to_MTNN(self, param_vector):
        """ Convert tensors in a ParamVector from network format to MTNN in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        @param param_vector The ParamVector to convert to MTNN format.
        """
        raise NotImplementedError

    
    @abstractmethod
    def convert_MTNN_format_to_network(self, param_vector):
        """ Convert tensors in a ParamVector from MTNN format to network format in place.

        The network format is the tensor format of the weight and bias
        tensors as used by the neural network. The MTNN format is
        the tensor format appropriate for numerical restriction and
        prolongation linear algebra based computation.

        @param param_vector The ParamVector to convert to neural network format.
        """
        raise NotImplementedError
