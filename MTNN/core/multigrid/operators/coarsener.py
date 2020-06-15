import numpy as np


class _HEMCoarsener():
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
        self.Fine2CoursePerLayer = None

    def HeavyEdgeMatching(self, similarityMatrix):
        """This function computes a heavy-edge-matching based on the similarity matrix S
        Constructs numColumnIn_list and the restrictionElementPos_list.
        These will be used in the restriction operators setup to construct the restriction matrix
        per layer per level.
        """
        n = similarityMatrix.shape[0]
        # Sort the rows of similarityMatrix in descending order
        P = np.argsort(-similarityMatrix, 1)
        # initialization
        match = np.empty(n, dtype = int)
        cnode = np.empty(n, dtype = int)
        match.fill(-1)
        cnode.fill(-1)
        n_pair = 0
        n_single = 0
        # randomized ordering
        seq = range(0, n)
        if self.randseq:
            seq = np.random.permutation(seq)
        sim = 0.0
        # greedy algorithm
        for i in seq:
            if match[i] == -1:
                for j in range(0, n):
                    c = P[i, j]
                    if (c != i and match[c] == -1 and similarityMatrix[i, c] > self.theta):
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
        fine2course = np.empty(n, dtype = int)
        fine2course.fill(-1)
        for i in range(n):
            if (fine2course[i] == -1):
                fine2course[i] = num_ColumnIn
                fine2course[match[i]] = num_ColumnIn
                num_ColumnIn += 1
        assert num_ColumnIn == n_pair + n_single
        # print('hem: %.2f' % (sim / n_pair))

        #print(fine2course, num_ColumnIn)
        return fine2course, num_ColumnIn

    def coarsen(self, fine_level_net):

        num_layers = len(fine_level_net.layers)
        # number columns
        self.coarseLevelDim = [fine_level_net.layers[0].in_features]

        # fine2coarse array
        self.Fine2CoursePerLayer = []
        # coarsen layers
        for layer_id in range(num_layers - 1):
            w = fine_level_net.layers[layer_id].weight.detach().numpy()
            b = fine_level_net.layers[layer_id].bias.detach().numpy().reshape(-1, 1)
            wb = np.concatenate([w, b], axis=1)
            nr = np.linalg.norm(wb, ord=2, axis=1, keepdims=True)
            wb = wb / nr
            # f-size (w.shape[0])
            nf = fine_level_net.layers[layer_id].out_features
            # similarity strength
            similarity = abs(wb @ np.transpose(wb))
            for i in range(nf):
                similarity[i, i] = 0

            # pdb.set_trace()

            f2c, nc = self.HeavyEdgeMatching(similarity)


            #
            self.coarseLevelDim.append(nc)
            self.Fine2CoursePerLayer.append(f2c)
            #
            # print('Coarsen layer %d: %d --> %d' % (layer_id, nF, nC))
        #

        self.coarseLevelDim.append(fine_level_net.layers[num_layers - 1].out_features)

        print(f"restriction Element Position{self.Fine2CoursePerLayer}, coarse_level dim {self.coarseLevelDim}")