"""
Holds functions to build multigrid levels to pass to the mulgrid scheme
"""
# standard
from typing import Type, List

# torch
import torch.nn as nn

# local
from MTNN.core.multigrid.operators import smoother, prolongation, restriction
from MTNN.core.alg import stopping
import MTNN.core.multigrid.scheme as mg

def build_uniform_levels(num_levels, presmoother, postsmoother, prolongation_operator,
                         restriction_operator, coarsegrid_solver, corrector=None) -> List[mg.Level]:
    """
    Constructs a uniform set of levels
    Args:

        num_levels: <int>
        presmoother: <core.multigrid.operators.smoother>
        postsmoother: <core.multigrid.operators.smoother>
        prolongation_operator: <core.multigrid.operators.prolongation>
        restriction_operator: <core.multigrid.operators.restriction>
        coarsegrid_solver: <core.multigrid.operators.smoother>
        stopping_criteria: <core.alg.stopping>
        loss_function: <torch.nn.modules.loss>

    Returns:
        set_of_levels: <list>  A list of <multigrid.scheme.Level> objects

    """
    set_of_levels = []

    for i in range(num_levels):
        aLevel = mg.Level(id = i,
                          presmoother = presmoother,
                          postsmoother = postsmoother,
                          prolongation = prolongation_operator,
                          restriction = restriction_operator,
                          coarsegrid_solver = coarsegrid_solver,
                          corrector = corrector)
        set_of_levels.append(aLevel)

    return set_of_levels


def build_vcycle_levels(num_levels: int, presmoother: smoother, postsmoother: smoother,
                        prolongation_operator: prolongation, restriction_operator: restriction,
                        coarsegrid_solver,  corrector=None) -> List[mg.Level]:
    """
    Constructs set of standard VCycle levels
    Args:
        num_levels: <int>
        presmoother: <core.multigrid.operators.smoother>
        postsmoother: <core.multigrid.operators.smoother>
        prolongation_operator: <core.multigrid.operators.prolongation>
        restriction_operator: <core.multigrid.operators.restriction>
        coarsegrid_solver: <core.multigrid.operators.smoother>
        corrector: <core.multigrid.operators.tau_corrector>

    Returns:
        set_of_levels: <list>  A list of <multigrid.scheme.Level> objects

    """

    set_of_levels = []
    for i in range(0, num_levels):
        if i < num_levels - 1:
            level = mg.Level(id = i,
                             presmoother = presmoother,
                             postsmoother = postsmoother,
                             prolongation = prolongation_operator,
                             restriction = restriction_operator,
                             coarsegrid_solver = coarsegrid_solver,
                             corrector = corrector)
            set_of_levels.append(level)

        else:  # Last Level
            level = mg.Level(id = i,
                             presmoother = None,
                             postsmoother = None,
                             prolongation = None,
                             restriction = None,
                             coarsegrid_solver = coarsegrid_solver,
                             loss_fn = loss_fn,
                             corrector = corrector)

            set_of_levels.append(level)

    return set_of_levels



