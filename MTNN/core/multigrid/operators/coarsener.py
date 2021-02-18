# standard
import torch
import numpy as np
import pdb

#local
from MTNN.utils import logger
log = logger.get_logger(__name__, write_to_file =True)

class StandardSimilarity:
    def __init__(self):
        pass

    def calculate_similarity(self, WB, model, layer_ind):
        return torch.mm(WB, torch.transpose(WB, dim0=0, dim1=1))

class HSimilarity:
    def __init__(self, U):
        self.U = U

    def calculate_similarity(self, WB, model, layer_id):
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
        

class HEMCoarsener():
    """
    Heavy Edge Matching Coarsener
    Takes a fine-level and constructs the coarsening layer to be used
    for building the restriction operator
    """
    def __init__(self, similarity_calculator):
        """
        Args:
            theta:
            randseq:
        """
        self.similarity_calculator = similarity_calculator
        self.coarseLevelDim = None
        self.Fine2CoarsePerLayer = None

    def get_heavyedgematching(self, similarityMatrix, random_seq=False):
        threshold = 0.0
        n = similarityMatrix.shape[0]
        P = torch.argsort(-similarityMatrix, 1)
        match = P[:,0] # Best match
        range_n = torch.tensor(range(n), device="cuda")

        # A node has a paired match if it is its match's match AND
        # their similarity is at least the threshold
        is_paired = (range_n == match[match]) * (similarityMatrix[range_n, match[range_n]] > threshold)

        # Try again for the loners
        for i in range(100):
            if torch.sum(is_paired) == n:
                break
            similarityMatrix[:, is_paired] = -1.0
            P = torch.argsort(-similarityMatrix, 1)
            match[~is_paired] = P[~is_paired,0]
            # Same test as above, but don't test similarity threshold
            # for those we already paired because we set those values
            # to -1
            is_paired = (range_n == match[match]) * (is_paired + (similarityMatrix[range_n, match[range_n]] > threshold))

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
        return fine2coarse, n_coarse
        
    def coarsen(self, fine_level_net):
        """
        Coarsens a model
        Args:
            fine_level_net:

        Returns:

        """
        num_layers = len(fine_level_net.layers)
        # number columns
        self.coarseLevelDim = [fine_level_net.layers[0].in_features]

        # fine2coarse array
        self.Fine2CoarsePerLayer = []

#        hidden_outputs = fine_level_net.all_hidden_forward(self.training_data)

        # coarsen layers
        for layer_id in range(num_layers - 1):
            w = fine_level_net.layers[layer_id].weight.detach()
            b = fine_level_net.layers[layer_id].bias.detach().reshape(-1, 1)
            wb = torch.cat([w, b], dim=1)
            nr = torch.norm(wb, p=2, dim=1, keepdim=True)
            wb = wb / nr
            # f-size (w.shape[0])
            nf = fine_level_net.layers[layer_id].out_features

            similarity = self.similarity_calculator.calculate_similarity(wb, fine_level_net, layer_id)

#             H = hidden_outputs[layer_id]
#             # print(H.shape)
#             # print(torch.mean(H, dim=0), torch.norm(H, dim=0), torch.std(H, dim=0))
#             print(torch.min(torch.norm(H, dim=0)))
#             if layer_id > 0:
#                 H = H - torch.mean(H, dim=0)                
#                 stdvec = torch.std(H, dim=0)
#                 stdvec[stdvec==0] = 1.0
#                 H = H / stdvec
#             ipm_size = H.shape[1] + 1
#             inner_product_mat = torch.empty((ipm_size, ipm_size), device="cuda:0")
#             inner_product_mat[-1,:] = 0
#             inner_product_mat[:,-1] = 0
#             inner_product_mat[-1,-1] = 1.0
#             inner_product_mat[:-1,:-1] = torch.mm(torch.transpose(H, dim0=0, dim1=1), H)
# #            print(inner_product_mat.shape, torch.norm(inner_product_mat))
#             # similarity strength
#             similarity = torch.mm(wb, torch.mm(inner_product_mat, torch.transpose(wb, dim0=0, dim1=1)))
# #            similarity = abs(torch.mm(wb, torch.transpose(wb, dim0=0, dim1=1)))
#             # print(inner_product_mat)
#             # print(similarity)
            similarity.fill_diagonal_(0)
            f2c, num_ColIn = self.get_heavyedgematching(similarity)
            print("Layer {} has {} coarse neurons".format(layer_id, num_ColIn))
            print()
            self.coarseLevelDim.append(num_ColIn)
            self.Fine2CoarsePerLayer.append(f2c)
            
        self.coarseLevelDim.append(fine_level_net.layers[num_layers - 1].out_features)

