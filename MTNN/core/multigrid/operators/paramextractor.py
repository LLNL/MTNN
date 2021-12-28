import copy
import torch
from MTNN.utils.datatypes import ParamVector
from MTNN.utils import deviceloader
from abc import abstractmethod

############################################
# Extractors
#############################################

class ParameterExtractor:
    """ParameterExtractor.

    This class can pull parameters of various kinds (parameters,
    momentum, gradients) from a network and then convert it into MTNN
    format for restriction/prolongation and other processing. The MTNN
    format is the tensor format appropriate for numerical restriction
    and prolongation linear algebra based computation.
    """

    def __init__(self, converter):
        self.converter = converter

    @abstractmethod
    def perform_initial_extraction(self, level, *argv):
        """Extract parameters from network without conversion to MTNN format."""        
        raise NotImplementedError

    def extract_from_network(self, level, *argv):
        """Pull parameters out of a network and convert to MTNN format.

        @param level the Level object from which to extract parameters.
        @argv Other generic arguments used in extraction.
        """
        param_vectors = self.perform_initial_extraction(level, *argv)
        if type(param_vectors) == tuple or type(param_vectors) == list:
            for pv in param_vectors:
                # TODO: Should this be an in place alteration? Or should it return a new ParamVector?
                self.converter.convert_network_format_to_MTNN(pv)
        else:
            self.converter.convert_network_format_to_MTNN(param_vectors)
        return param_vectors

class ParamMomentumExtractor(ParameterExtractor):
    """ParameterExtractor.

    This class extracts parameter and momentum tensors from a neural
    network to a ParamVector for restriction/prolongation processing.

    """
    def __init__(self, converter):
        super().__init__(converter)

    def perform_initial_extraction(self, level):
        """Extract parameter and momentum tensors from a neural network.

        Note that this function doesn't depend on the type of network
        aside from it having a sequence of layers each of which
        contains a weight and bias tensor.

        """
        net = level.net
        optimizer = level.smoother.optimizer

        # Pull parameters from the network
        W_array = [net.layers[layer_id].weight.detach() for layer_id in range(len(net.layers))]
        B_array = [net.layers[layer_id].bias.detach().reshape(-1, 1) for layer_id in range(len(net.layers))]

        # Pull momentum values from the optimizer. This is a bit ugly,
        # but PyTorch wasn't expecting us to want this directly so
        # they didn't make it accessible in a nice way.
        get_p = lambda ind : optimizer.state[optimizer.param_groups[0]['params'][ind]]['momentum_buffer']
        mW_array, mB_array = [list(x) for x in zip(*[(get_p(2*i), get_p(2*i+1).reshape(-1, 1)) for i in
                                                     range(int(len(optimizer.param_groups[0]['params']) / 2))])]
        param_library, momentum_library = (ParamVector(W_array, B_array), ParamVector(mW_array, mB_array))
        return param_library, momentum_library
    
    def insert_into_network(self, level, param_library, momentum_library):
        """Insert ParamVector tensors into a network.

        @param level <Level> The level into which to insert the parameters.
        @param param_library <ParamVector> The tensors of network parameters to insert.
        @param momentum_library <ParamVector> The tensors of momentum values to insert.

        """
        # TODO: This function uses two parameter copies which is
        # unnecessary. Refactor to only use one.

        # Create copy of initial coarse values. Needed during
        # prolongation to compute differences.
        level.init_params = copy.deepcopy(param_library)
        level.init_momentum = copy.deepcopy(momentum_library)

        self.converter.convert_MTNN_format_to_network(param_library)
        self.converter.convert_MTNN_format_to_network(momentum_library)
        W_array, B_array = param_library.weights, param_library.biases
        mW_array, mB_array = momentum_library.weights, momentum_library.biases

        # Insert parameters into the network
        with torch.no_grad():
            for layer_id in range(len(level.net.layers)):
                level.net.layers[layer_id].weight.copy_(W_array[layer_id])
                level.net.layers[layer_id].bias.copy_(B_array[layer_id].reshape(-1))
        level.net.zero_grad()

        # Insert momemntum values into the optimizer
        momentum_data = []
        with torch.no_grad():
            for i in range(len(mW_array)):
                momentum_data.append(mW_array[i].clone())
                momentum_data.append(mB_array[i].clone().reshape(-1))
        level.smoother.set_momentum(momentum_data)
        level.smoother.optimizer = None

    def add_to_network(self, level, param_diff_library, momentum_diff_library):
        """Add ParamVector difference value tensors to a network.

        This is similar to insert_into_network, except that it does not
        replace the values of the network but instead adds the inputs
        to the existing values. This is done, for example, in a Full
        Approximation Scheme coarse-grid correction.

        Inputs:
        @param level <Level> The level into which to insert the parameters.
        @param param_diff_library <ParamVector> The tensors of network parameters to insert.
        @param momentum_diff_library <ParamVector> The tensors of momentum values to insert.

        """
        self.converter.convert_MTNN_format_to_network(param_diff_library)
        self.converter.convert_MTNN_format_to_network(momentum_diff_library)
        dW_array, dB_array = param_diff_library.weights, param_diff_library.biases
        dmW_array, dmB_array = momentum_diff_library.weights, momentum_diff_library.biases

        optimizer = level.smoother.optimizer
        get_p = lambda ind : optimizer.state[optimizer.param_groups[0]['params'][ind]]['momentum_buffer']

        with torch.no_grad():
            for layer_id in range(len(level.net.layers)):
                # Add parameters to the network
                level.net.layers[layer_id].weight.add_(dW_array[layer_id])
                level.net.layers[layer_id].bias.add_(dB_array[layer_id].reshape(-1))
                # Add momentum values to the optimizer.
                get_p(2*layer_id).add_(dmW_array[layer_id])
                get_p(2*layer_id+1).add_(dmB_array[layer_id].reshape(-1))


class GradientExtractor(ParameterExtractor):
    """GradientExtractor.

    Given a dataloder and a loss function, this class extracts
    gradient values from the neural network associated with the
    examples in the dataloder. This is used during the computation of
    the tau correction.

    """
    def __init__(self, converter):
        super().__init__(converter)

    def perform_initial_extraction(self, level, dataloader, loss_fn):
        """Extract gradient values associated with the training examples in a
        dataloder. Used in the tau correction.

        """
        # Compute the gradient
        net = level.net
        net.zero_grad()   # zero the parameter gradient
        total_loss = None
        for i, mini_batch_data in enumerate(dataloader, 0):
            # get the inputs. data is a list of [inputs, labels]
            inputs, labels = deviceloader.load_data(mini_batch_data, net.device)

            # forward: get the loss w.r.t this batch
            outputs = net(inputs)

            if total_loss is None:
                total_loss = loss_fn(outputs, labels)
            else:
                total_loss += loss_fn(outputs, labels)

        W_grad_array, B_grad_array = [], []
        if total_loss is None:
            # If this was an empty dataloader, total_loss will still be
            # None, so grad should be all 0s
            for layer_id in range(len(net.layers)):
                W_grad_array.append(torch.zeros(net.layers[layer_id].weight.shape).to(level.net.device))
                B_grad_array.append(torch.zeros(net.layers[layer_id].bias.shape).reshape(-1, 1).to(level.net.device))
        else:
            # dataloader wasn't empty, so have a nontrivial gradient
            # to compute.
            total_loss.backward()
            for layer_id in range(len(net.layers)):
                W_grad_array.append(net.layers[layer_id].weight.grad.detach().clone())
                B_grad_array.append(net.layers[layer_id].bias.grad.detach().reshape(-1, 1).clone())

        return ParamVector(W_grad_array, B_grad_array)

