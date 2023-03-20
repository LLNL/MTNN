"""
Algorithms to calculate similarity between layers
"""
# standard
import torch

#local
from MTNN.utils import logger
from MTNN.utils import deviceloader
from MTNN.utils.datatypes import CoarseMapping

# Public
__all__ = ['StandardSimilarity',
           'HSimilarity',
           'ExpHSimilarity',
           'RadialBasisSimilarity',
           'HEMCoarsener']

class StandardSimilarity:
    """Computes a similarity matrix using the standard inner product
    between neuron parameter vectors.

    That is, $sim(i,j) = w_i^T w_j.$

    """
    def __init__(self):
        pass

    def calculate_similarity(self, WB, model, layer_ind):
        """ Calculate similarity.

        @param WB The concatenation of a weight matrix with a bias vector into a single parameter matrix.
        @param model (Unused)
        @param layer_ind (Unused)
        """
        nr = torch.norm(WB, p=2, dim=1, keepdim=True)
        WB = WB / nr
        return torch.mm(WB, torch.transpose(WB, dim0=0, dim1=1))

class RadialBasisSimilarity:
    """Compute the radial basis similarity between neurons. Note that this
    is a valid inner product similarity via the kernel trick. See,
    e.g. www.wikipedia.com/radial_basis_kernel

    That is,
    sim(i,j) = exp(-\lambda ||w_i - w_j||^2) = exp(-lambda ((w_i, w_i) + (w_j, w_j) - 2 (w_i, w_j)))

    """
    def __init__(self, scale=1.0):
        self.scale = scale

    def calculate_similarity(self, WB, model, layer_id):
        """ Calculate similarity.

        @param WB The concatenation of a weight matrix with a bias vector into a single parameter matrix.
        @param model (Unused)
        @param layer_ind (Unused)
        """
        S = torch.mm(WB, torch.transpose(WB, dim0=0, dim1=1))
        D = torch.diag(S) * torch.ones(S.shape)
        to_ret = torch.exp(self.scale * (2 * S - D - D.T))
        return to_ret
    

class HEMMatcher():
    """Heavy Edge Matching

    Given a ParamVector of fine-level tensors, computes a matching
    between neurons (or channels in convolutional layers) such that
    matched neurons should be merged together during restriction.

    """
    def __init__(self, similarity_calculator, coarsen_on_layer = None, num_retries = 100):
        """Constructor

        @param similarity_calculator <Class> Class that measures
        similarity between two neurons.
            
        @param coarsen_on_layer <list[bool] or None> For each layer,
        if False, return identity matching. If arg is None, coarsen
        every layer.

        @param num_retries Number of times to retry finding a match for unmatched neurons.

        """
        self.similarity_calculator = similarity_calculator
        self.coarseLevelDim = None
        self.Fine2CoarsePerLayer = None
        self.coarsen_on_layer = coarsen_on_layer
        self.num_retries = num_retries
        self.log = logger.get_MTNN_logger()

    def get_heavyedgematching(self, similarityMatrix):
        """Compute matching for a single layer.

        @param similarityMatrix A num_neurons x num_neurons symmetric
        matrix containing similarity values.

        Return values:
        match : An array containing, for each neurons, the index its match.
        n_coarse : The number of computed coarse neurons.
        """
        threshold = 0.0
        n = similarityMatrix.shape[0]
        P = torch.argsort(-similarityMatrix, 1)

        # Each neuron wants to be with its highest-similarity neighbor.
        match = P[:,0]

        range_n = torch.tensor(range(n), device=deviceloader.get_device())

        # A node has a paired match if it is its match's match AND
        # their similarity is at least the threshold
        is_paired = (range_n == match[match]) * (similarityMatrix[range_n, match[range_n]] > threshold)

        # Try again for the loners
        last_num_paired = -1
        for i in range(self.num_retries):
            curr_num_paired = torch.sum(is_paired)
            if curr_num_paired == n or curr_num_paired == last_num_paired:
                break
            last_num_paired = curr_num_paired
            similarityMatrix[:, is_paired] = -1.0
            P = torch.argsort(-similarityMatrix, 1)
            match[~is_paired] = P[~is_paired,0]
            # Same test as above, but don't test similarity threshold
            # for those we already paired because we set those values
            # to -1
            is_paired = (range_n == match[match]) * (is_paired + (similarityMatrix[range_n, match[range_n]] > threshold))

        # Unpaired neurons get matched to themselves
        match[~is_paired] = range_n[~is_paired]

        n_coarse = int(n - (torch.sum(is_paired) / 2))
        return match, n_coarse
        
    def __call__(self, param_vector, net):
        """Compute the matching between represented in the ParamVector.

        @param param_vector The set of neuron parameters to match up.

        @param net A neural network, useful in some cases for
        computing similarity. (TODO: Replace net with more generic
        *argv)

        """
        W_array, B_array = param_vector.weights, param_vector.biases
        
        num_layers = len(W_array)
        # number columns
        num_coarse_array = []

        # neuron matchings. another way of writing the same information as in fine2CoarsePerLayer
        match_per_layer = []

        # coarsen layers
        for layer_id in range(num_layers - 1):
            # If a higher-order tensor, flatten into a matrix. TODO:
            # Push the flattening into the similarity functions to
            # enable better extensibility.
            w = W_array[layer_id].transpose(0, -2).flatten(1)
            b = B_array[layer_id]
            wb = torch.cat([w, b], dim=1)

            # f-size (w.shape[0])
            nf = wb.shape[0]

            if self.coarsen_on_layer is None or self.coarsen_on_layer[layer_id]:
                similarity = self.similarity_calculator.calculate_similarity(wb, net, layer_id)

                similarity.fill_diagonal_(-999.0)
                match, num_ColIn = self.get_heavyedgematching(similarity)
                num_coarse_array.append(num_ColIn)
                match_per_layer.append(match)
            else:
                num_coarse_array.append(nf)
            self.log.info("Layer {} coarsened to {} coarse neurons".format(layer_id, num_coarse_array[-1]))

        return CoarseMapping(num_coarse_array, match_per_layer)

