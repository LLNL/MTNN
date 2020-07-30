import numpy as np
import collections as col
# local
import MTNN.core.multigrid.scheme as mg
import MTNN.core.multigrid.operators._coarsener as coarsener
import MTNN.utils.logger as log

log = log.get_logger(__name__, write_to_file = True)


class PairwiseAggCoarsener:
    """
    Sets up intermediate values for Pairwise Aggregation-based Restriction and Prolongation Operators
        * Uses Heavy Edge matching from similarity matrix of source_model's Weight and Bias
    """

    def __init__(self):
        # A list of restriction operators(matrices) to use
        # for each fine_level network's hidden layer



        self.coarsener = coarsener._HEMCoarsener()
        #self.prolongation_operators = None # a list of arrays, each corresponding to a hidden layer
        #self.restriction_operators = None
        #self.interpolation_data = None

    def setup(self, fine_level: mg.Level, coarse_level: mg.Level):
        """
        Construct a restriction and prolongation operators per layer of the fine_level.net
        based on Heavy Edge Matching Coarsener
        Args:
            fine_level <MTNN.core.multigrid.scheme.Level>
            coarse_level < MTNN.core.multigrid.scheme.Level>
        Returns:
            self.restriction_operators <list> of matrix operators to apply to the fine level net to get a coarse level net.
            self.prolongation_operators <list> of matrix operators to apply to the coarse level net to get a fine level net
        """
        # Each index of Prolongation/Restriction operators corresponds to layer to which it is to be applied
        prolongation_operators = []
        restriction_operators = []

        # Instantiate the coarse-level net with the coarsener dimensions
        self.coarsener.coarsen(fine_level.net)

#        log.debug(f"interpolator.setup: Coarse Level Net dimensions: {self.coarsener.coarseLevelDim}")
        coarse_level.net = fine_level.net.__class__(self.coarsener.coarseLevelDim,
                                                    fine_level.net.activation,
                                                    fine_level.net.output)
#        log.debug(f"interpolator.setup {coarse_level.net = }")

        # Create the restriction operator per hidden layer (except for the last layer)
        # from the coarsener
        num_layers = len(fine_level.net.layers)
        for layer_id in range(num_layers - 1):

            original_W = np.copy(fine_level.net.layers[layer_id].weight.detach().numpy())
            original_b = np.copy(fine_level.net.layers[layer_id].bias.detach().numpy().reshape(-1, 1))

            WB = np.concatenate([original_W, original_b], axis = 1)
            nr = np.linalg.norm(WB, ord = 2, axis = 1, keepdims = True)

            F2C_layer = self.coarsener.Fine2CoarsePerLayer[layer_id]

            nF = fine_level.net.layers[layer_id].out_features
            nC = coarse_level.net.layers[layer_id].out_features
            R_layer = np.zeros([nC, nF])

            # Construct restriction operators for each layer
            for i in range(nF):
                # print(f"Constructing R: {i}, {F2C_l[i]}")
                R_layer[F2C_layer[i], i] = 1
            # d = diag(R*B*R^T)
            d = (R_layer * nr.reshape(1, -1) @ np.transpose(R_layer)).diagonal().reshape(-1, 1)
            # P = diag(nr) * R^T * diag(1./d)

            restriction_operators.append(R_layer)

            # Construct prolongation operators for each layer
            # TODO - Decouple?
            P_layer = np.transpose(R_layer) * nr / d.reshape(1, -1)
            prolongation_operators.append(P_layer)

#           log.info(f"\n R {R_layer} \n P {P_layer}")
            # pdb.set_trace()

#        log.debug(f"Coarsener.setup: \n{restriction_operators=} \n{prolongation_operators=}")
        interpolation_data = col.namedtuple("interpolation_data", "R_op P_op")
        return interpolation_data(restriction_operators, prolongation_operators)

