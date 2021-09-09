# standard
import torch
import numpy as np
import pdb

#local
from MTNN.utils import logger
from MTNN.utils import deviceloader
from MTNN.utils.datatypes import CoarseMapping
log = logger.get_logger(__name__, write_to_file =True)

class StandardSimilarity:
    """ sim(i,j) = w_i^T w_j
    """
    def __init__(self):
        pass

    def calculate_similarity(self, WB, model, layer_ind):
        nr = torch.norm(WB, p=2, dim=1, keepdim=True)
        WB = WB / nr
        return torch.mm(WB, torch.transpose(WB, dim0=0, dim1=1))

class HSimilarity:
    """
    Like StandardSimilarity, but instead of comparing vectors
    direclty, we feed a set of training data U through the network,
    and compare the output vectors of each internal layer.
    """
    def __init__(self, U):
        self.U = U

    def calculate_similarity(self, WB, model, layer_id):
        nr = torch.norm(WB, p=2, dim=1, keepdim=True)
        WB = WB / nr
        if layer_id == 0:
            self.hidden_outputs = model.all_hidden_forward(self.U)
            H = self.hidden_outputs[layer_id]
        else:
            H = self.hidden_outputs[layer_id]
            H = H - torch.mean(H, dim=0)                
            normvec = torch.norm(H, dim=0)
            normvec[normvec==0] = 1.0
            H = H / normvec
        ipm_size = H.shape[1] + 1
        inner_product_mat = torch.empty((ipm_size, ipm_size), device="cuda:0")
        inner_product_mat[-1,:] = 0
        inner_product_mat[:,-1] = 0
        inner_product_mat[-1,-1] = 1.0
        inner_product_mat[:-1,:-1] = torch.mm(torch.transpose(H, dim0=0, dim1=1), H)
        return torch.mm(WB, torch.mm(inner_product_mat, torch.transpose(WB, dim0=0, dim1=1)))

class ExpHSimilarity:
    def __init__(self, U, scale=1.0):
        self.U = U
        self.scale = scale

    def calculate_similarity(self, WB, model, layer_id):
        nr = torch.norm(WB, p=2, dim=1, keepdim=True)
        WB = WB / nr
        if layer_id == 0:
            self.hidden_outputs = model.all_hidden_forward(self.U)
            H = self.hidden_outputs[layer_id]
        else:
            H = self.hidden_outputs[layer_id]
            H = H - torch.mean(H, dim=0)                
            stdvec = torch.std(H, dim=0)
            stdvec[stdvec==0] = 1.0
            H = H / stdvec
        ipm_init = torch.mm(torch.transpose(H, dim0=0, dim1=1), H)
        D = torch.diag(ipm_init) * torch.ones(ipm_init.shape)
        eS = torch.exp(self.scale * (2 * S - D - D.T))

        ipm_size = H.shape[1] + 1
        inner_product_mat = torch.empty((ipm_size, ipm_size), device="cuda:0")
        inner_product_mat[-1,:] = 0
        inner_product_mat[:,-1] = 0
        inner_product_mat[-1,-1] = 1.0
        inner_product_mat[:-1,:-1] = eS
        return torch.mm(WB, torch.mm(inner_product_mat, torch.transpose(WB, dim0=0, dim1=1)))

class RadialBasisSimilarity:
    """ sim(i,j) = exp(-\lambda ||w_i - w_j||^2) = exp(-lambda ((w_i, w_i) + (w_j, w_j) - 2 (w_i, w_j)))
    """
    def __init__(self, scale=1.0):
        self.scale = scale

    def calculate_similarity(self, W, B, model, layer_id):
        S = torch.mm(WB, torch.transpose(WB, dim0=0, dim1=1))
        D = torch.diag(S) * torch.ones(S.shape)
        to_ret = torch.exp(self.scale * (2 * S - D - D.T))
        return to_ret
    

class HEMCoarsener():
    """
    Heavy Edge Matching Coarsener
    Takes a fine-level and constructs the coarsening layer to be used
    for building the restriction operator
    """
    def __init__(self, similarity_calculator, coarsen_on_layer = None):
        """
        Args:
            similarity_calculator (class) Class that measures similarity between two neurons.
            coarsen_on_layer (array[bool]) For each layer, if False, return identity matching. 
                                           If arg is None, coarsen every layer.
        """
        self.similarity_calculator = similarity_calculator
        self.coarseLevelDim = None
        self.Fine2CoarsePerLayer = None
        self.coarsen_on_layer = coarsen_on_layer

    def get_heavyedgematching(self, similarityMatrix, random_seq=False):
        threshold = 0.0
        n = similarityMatrix.shape[0]
        P = torch.argsort(-similarityMatrix, 1)
        match = P[:,0] # Best match
        range_n = torch.tensor(range(n), device=deviceloader.get_device())

        # A node has a paired match if it is its match's match AND
        # their similarity is at least the threshold
        is_paired = (range_n == match[match]) * (similarityMatrix[range_n, match[range_n]] > threshold)

        # Try again for the loners
        last_num_paired = -1
        for i in range(100):
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

        # fine2coarse = -1
        # n_coarse = int(n - (torch.sum(is_paired) / 2))

        # fine2coarse = torch.full([n], -1, dtype=int)
        # curr_coarse = 0
        # for i in range(n):
        #     if match[i] < i:
        #         fine2coarse[i] = fine2coarse[match[i]]
        #     else:
        #         fine2coarse[i] = curr_coarse
        #         curr_coarse += 1
        # # n_coarse = curr_coarse

        # For each paired match, choose a "base" which is by
        # convention the fine neuron with lower index.
        bases_of_pairs = torch.unique(torch.min(match[is_paired], match[match[is_paired]]))
        n_pair = len(bases_of_pairs)

        fine2coarse = torch.full([n], -1, dtype=int)
        # Each base is associated with a coarse neuron index
        fine2coarse[bases_of_pairs] = torch.tensor(range(n_pair), dtype=int)
        # Each match is associated with the same coarse index as its base.
        fine2coarse[match[bases_of_pairs]] = fine2coarse[bases_of_pairs]

        # Each singleton gets its own coarse neuron.
        fine2coarse[~is_paired] = n_pair + torch.tensor(range(n - 2 * n_pair), dtype=int)

        n_coarse = n - n_pair # = n_pair + (n - 2 * n_pair)
        return match, fine2coarse, n_coarse
        
    def __call__(self, param_matrix_list, net):
        """
        """
        W_array, B_array = param_matrix_list.weights, param_matrix_list.biases
        
        num_layers = len(W_array)
        # number columns
        num_coarse_array = []

        # fine2coarse array
        fine2CoarsePerLayer = []

        # neuron matchings. another way of writing the same information as in fine2CoarsePerLayer
        match_per_layer = []

        # coarsen layers
        for layer_id in range(num_layers - 1):
            w = W_array[layer_id].transpose(0, -2).flatten(1)
            b = B_array[layer_id]
            wb = torch.cat([w, b], dim=1)

            # f-size (w.shape[0])
            nf = wb.shape[0]

            if self.coarsen_on_layer is None or self.coarsen_on_layer[layer_id]:
                similarity = self.similarity_calculator.calculate_similarity(wb, net, layer_id)

                similarity.fill_diagonal_(-999.0)
                match, fine2coarse, num_ColIn = self.get_heavyedgematching(similarity)
                num_coarse_array.append(num_ColIn)
                fine2CoarsePerLayer.append(fine2coarse)
                match_per_layer.append(match)
            else:
                num_coarse_array.append(nf)
                fine2CoarsePerLayer.append(torch.arange(0, nf, dtype=int))
            print("Layer {} has {} coarse neurons".format(layer_id, num_coarse_array[-1]))
            print()

        return CoarseMapping(fine2CoarsePerLayer, num_coarse_array, match_per_layer)

