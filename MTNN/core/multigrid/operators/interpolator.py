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

def transfer(Wmats, Bmats, R_ops, P_ops):
    num_layers = len(Wmats)
    Wdest_array = []
    Bdest_array = []
    for layer_id in range(num_layers):
        Wsrc = Wmats[layer_id]
        Bsrc = Bmats[layer_id]
        if layer_id < num_layers - 1:
            if layer_id == 0:
                Wdest = R_ops[layer_id] @ Wsrc
            else:
                Wdest = R_ops[layer_id] @ Wsrc @ P_ops[layer_id - 1]
            Bdest = R_ops[layer_id] @ Bsrc
        elif layer_id > 0:
            Wdest = Wsrc @ P_ops[layer_id-1]
            Bdest = Bsrc.clone()

        Wdest_array.append(Wdest)
        Bdest_array.append(Bdest)
    return Wdest_array, Bdest_array

def transfer_star(Wmats, Bmats, R_ops, P_ops):
    num_layers = len(Wmats)
    Wdest_array = []
    Bdest_array = []
    for layer_id in range(num_layers):
        Wsrc = Wmats[layer_id]
        Bsrc = Bmats[layer_id]
        if layer_id < num_layers - 1:
            if layer_id == 0:
                Wdest = R_ops[layer_id] * Wsrc
            else:
                Wdest = R_ops[layer_id] * Wsrc * P_ops[layer_id - 1]
            Bdest = R_ops[layer_id] * Bsrc
        elif layer_id > 0:
            Wdest = Wsrc * P_ops[layer_id-1]
            Bdest = Bsrc.clone()

        Wdest_array.append(Wdest)
        Bdest_array.append(Bdest)
    return Wdest_array, Bdest_array

class PairwiseAggCoarsener:
    """
    Sets up intermediate values for Pairwise Aggregation-based Restriction and Prolongation Operators
        * Uses Heavy Edge matching from similarity matrix of source_model's Weight and Bias
    """

    def __init__(self, coarsener):
        self.coarsener = coarsener


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
        l2reg_left_vecs = []
        l2reg_right_vecs = []

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

            # ========================================================================
            # Compute matrix A s.t. l2 regularization on coarse level uses x_c^T A x_c
            # But we compute vectors used in the action of A, not A itself.
            # ========================================================================
            # l2reg_left = torch.diag(R_for_grad_layer @ P_layer).reshape(-1, 1)
            # l2reg_left_vecs.append(l2reg_left)
            # l2reg_right = torch.diag(R_layer @ P_for_grad_layer).reshape(1, -1)
            # l2reg_right_vecs.append(l2reg_right)

        return mgdata.operators(restriction_operators, prolongation_operators, restriction_for_grad_operators,
                                prolongation_for_grad_operators, l2reg_left_vecs, l2reg_right_vecs)

