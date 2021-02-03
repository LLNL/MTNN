# standard
import torch
import numpy as np
import pdb

#local
from MTNN.utils import logger
log = logger.get_logger(__name__, write_to_file =True)

class HEMCoarsener():
    """
    Heavy Edge Matching Coarsener
    Takes a fine-level and constructs the coarsening layer to be used
    for building the restriction operator
    """
    def __init__(self, theta=0.0, randseq=False):
        """
        Args:
            theta:
            randseq:
        """
        self.theta = theta
        self.randseq = randseq
        self.coarseLevelDim = None
        self.Fine2CoarsePerLayer = None


    def get_heavyedgematching(self, similarityMatrix, random_seq=False):
        """This function computes a heavy-edge-matching based on the similarity matrix S
        Constructs numColumnIn_list and the restrictionElementPos_list.
        These will be used in the restriction operators setup to construct the restriction matrix
        per layer per level.
        Args:
            similarityMatrix: <numpy matrix>
            random_seq: <bool> Enables randomized ordering of rows
        """

        n = similarityMatrix.shape[0]
        # Sort the rows of similarityMatrix in descending order
        P = torch.argsort(-similarityMatrix, 1)
        # initialization
        cnode = torch.full(torch.Size([n]), -1, dtype=int)
        match = torch.full(torch.Size([n]), -1, dtype=int)
        n_pair, n_single = 0, 0
        # randomized ordering
        seq = range(0, n)
        if random_seq:
            seq = torch.randperm(seq)
        sim = 0.0
        for i in seq:
            if match[i] == -1:
                for j in range(0, n):
                    c = P[i, j]
                    if (c != i and match[c] == -1 and similarityMatrix[i, c] > 0):
                        match[i] = c
                        match[c] = i
                        cnode[i] = cnode[c] = n_pair + n_single
                        n_pair += 1
                        sim += similarityMatrix[i, c]
                        break
                if match[i] == -1:
                    match[i] = i
                    cnode[i] = n_pair + n_single
                    n_single += 1
        # Column-size/ In-features per layer
        num_ColumnIn= 0
        # Construct the fine
        fine2course = torch.full(torch.Size([n]), -1, dtype=int)
        for i in range(n):
            if (fine2course[i] == -1):
                fine2course[i] = num_ColumnIn
                fine2course[match[i]] = num_ColumnIn
                num_ColumnIn += 1
        assert num_ColumnIn == n_pair + n_single
        # print('hem: %.2f' % (sim / n_pair))
        return fine2course, num_ColumnIn

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

        # coarsen layers
        for layer_id in range(num_layers - 1):
            w = fine_level_net.layers[layer_id].weight.detach()
            b = fine_level_net.layers[layer_id].bias.detach().reshape(-1, 1)
            wb = torch.cat([w, b], dim=1)
            nr = torch.norm(wb, p=2, dim=1, keepdim=True)
            wb = wb / nr
            # f-size (w.shape[0])
            nf = fine_level_net.layers[layer_id].out_features
            # similarity strength
            similarity = abs(torch.mm(wb, torch.transpose(wb, dim0=0, dim1=1)))
            for i in range(nf):
                similarity[i, i] = 0
            f2c, num_ColIn = self.get_heavyedgematching(similarity)
            self.coarseLevelDim.append(num_ColIn)
            self.Fine2CoarsePerLayer.append(f2c)
            

        self.coarseLevelDim.append(fine_level_net.layers[num_layers - 1].out_features)

