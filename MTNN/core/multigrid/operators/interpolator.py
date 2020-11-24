# standard 
import collections as col
# torch 
import torch
# local
import MTNN.core.multigrid.scheme as mg
import MTNN.core.multigrid.operators.coarsener as coarsener
import MTNN.utils.logger as log
import MTNN.utils.datatypes as mgdata

log = log.get_logger(__name__, write_to_file = True)


class PairwiseAggCoarsener:
    """
    Sets up intermediate values for Pairwise Aggregation-based Restriction and Prolongation Operators
        * Uses Heavy Edge matching from similarity matrix of source_model's Weight and Bias
    """

    def __init__(self):
        # A list of restriction operators(matrices) to use
        # for each fine_level network's hidden layer
        self.coarsener = coarsener.HEMCoarsener()


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
        prolongation_for_grad_operators = []
        restriction_for_grad_operators = []

        # Instantiate the coarse-level net with the coarsener dimensions
        self.coarsener.coarsen(fine_level.net)

        coarse_level.net = fine_level.net.__class__(self.coarsener.coarseLevelDim,
                                                    fine_level.net.activation,
                                                    fine_level.net.output)
        coarse_level.net.set_device(fine_level.net.device)

        # Create the restriction operator per hidden layer (except for the last layer)
        # from the coarsener

        num_layers = len(fine_level.net.layers)
        for layer_id in range(num_layers - 1):
            
            original_W = fine_level.net.layers[layer_id].weight.detach().clone()
            original_b = fine_level.net.layers[layer_id].bias.detach().clone().reshape(-1, 1)

            WB = torch.cat((original_W, original_b), dim=1)
            nr = torch.norm(WB, p=2, dim=1, keepdim=True)

            F2C_layer = self.coarsener.Fine2CoarsePerLayer[layer_id]

            nF = fine_level.net.layers[layer_id].out_features
            nC = coarse_level.net.layers[layer_id].out_features
            R_layer = torch.zeros([nC, nF]).to(fine_level.net.device)

            # Construct restriction operators for each layer
            for i in range(nF):
                # print(f"Constructing R: {i}, {F2C_l[i]}")
                R_layer[F2C_layer[i], i] = 1
            # d = diag(R*B*R^T)
           
            d = (R_layer * torch.reshape(nr, (1, -1)) @ torch.transpose(R_layer, 0, 1)).diagonal().reshape(-1, 1)
            #d = torch.mm(R_layer * torch.reshape(nr, (1, -1)), torch.transpose(R_layer, 0, 1).diagonal().reshape(-1,1))
            
            # P = diag(nr) * R^T * diag(1./d)
            restriction_operators.append(R_layer)

            # Construct prolongation operators for each layer
            # TODO - Decouple?
            P_layer = (torch.transpose(R_layer, 0, 1) * nr) / d.reshape(1, -1)
            prolongation_operators.append(P_layer)

            R_for_grad_layer = torch.transpose(P_layer, 0, 1)
            restriction_for_grad_operators.append(R_for_grad_layer)

            P_for_grad_layer = torch.transpose(R_layer, 0, 1)
            prolongation_for_grad_operators.append(P_for_grad_layer)

        return mgdata.operators(restriction_operators, prolongation_operators, restriction_for_grad_operators,
                                prolongation_for_grad_operators)

