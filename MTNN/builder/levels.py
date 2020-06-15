"""
Holds functions to build multigrid levels to pass to the mulgrid scheme
"""

import MTNN.core.multigrid.scheme as mg


def build_uniform_levels(num_levels=0, presmoother=None, postsmoother=None, prolongation_operator=None,
                         restriction_operator=None, coarsegrid_solver=None, stopping_criteria=None):
    """
    Constructs a uniform set of levels
    Args:
        num_levels:
        presmoother:
        postsmoother:
        prolongation_operator:
        restriction_operator:
        coarsegrid_solver:
        stopping_criteria:

    Returns:
        set_of_levels: <list> A list of multigrid.scheme.Level Objects

    """
    set_of_levels = []

    for num in range(num_levels):
        aLevel = mg.Level(presmoother, postsmoother, prolongation_operator,
                          restriction_operator, coarsegrid_solver, stopping_criteria)
        set_of_levels.append(aLevel)

    return set_of_levels


def build_vcycle_levels(num_levels:int, presmoother, postsmoother,
                 prolongation_operator, restriction_operator,coarsegrid_solver,
                 stopper):
    """
    Constructs set of standard VCycle levels
    Args:
        num_levels:
        presmoother:
        postsmoother:
        prolongation_operator:
        restriction_operator:
        coarsegrid_solver:
        stopper:

    Returns:

    """

    set_of_levels = []


    for i in range(0, num_levels):
        if i < num_levels - 1:
            level = mg.Level(presmoother = presmoother,
                             postsmoother = postsmoother,
                             prolongation = prolongation_operator,
                             restriction = restriction_operator,
                             coarsegrid_solver = coarsegrid_solver,
                             stopping_measure = stopper)
            set_of_levels.append(level)
        else:
            level = mg.Level(presmoother = None,
                             postsmoother = None,
                             prolongation = None,
                             restriction = None,
                             coarsegrid_solver = coarsegrid_solver,
                             stopping_measure = stopper)

            set_of_levels.append(level)

    return set_of_levels