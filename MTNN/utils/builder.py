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

#TODO: Fix new instances on each level!

def build_uniform_levels(num_levels=0, presmoother=None, postsmoother=None, prolongation_operator=None,
                         restriction_operator=None, coarsegrid_solver=None, stopping_criteria=None,
                         loss_function=None) -> List[mg.Level]:
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

    for i in range(num_levels):
        aLevel = mg.Level(id = i,
                          presmoother = presmoother,
                          postsmoother = postsmoother,
                          prolongation = prolongation_operator,
                          restriction = restriction_operator,
                          coarsegrid_solver = coarsegrid_solver,
                          stopping_measure = stopping_criteria,
                          loss_fn = loss_function)
        set_of_levels.append(aLevel)

    return set_of_levels


def build_vcycle_levels(num_levels:int, presmoother: smoother, postsmoother: smoother,
                        prolongation_operator: prolongation, restriction_operator: restriction,
                        coarsegrid_solver, stopper: stopping._BaseStopper,
                        loss_function: Type[nn.modules.loss._Loss]) -> List[mg.Level]:
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
        loss_function:

    Returns:
        set_of_levels: <list> of MTNN.core.multigrid.scheme.level objects

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
                             stopping_measure = stopper,
                             loss_fn = loss_function)
            set_of_levels.append(level)

        else:  # Last Level
            level = mg.Level(id = i,
                             presmoother = None,
                             postsmoother = None,
                             prolongation = None,
                             restriction = None,
                             coarsegrid_solver = coarsegrid_solver,
                             stopping_measure = stopper,
                             loss_fn = loss_function)

            set_of_levels.append(level)

    return set_of_levels



"""WIP
def build_stoppers(stopperTypes: List[Type[stopping._BaseStopper]],
                   stopping_measure: list):

    assert len(stopperTypes) == len(stopping_measure)
    set_of_stoppers = []

    for stopper, arg in zip(stopperTypes, stopping_measure):
        try:
            print(stopper)
            stopper(arg)

        except Exception as e:
            raise e
    return set_of_stoppers


def build_smoothers(smootherTypes: List[Type[smoother._BaseSmoother]],
                   stopperTypes: List[Type[stopping._BaseStopper]],
                   stopping_measure: list):

    for stopper in build_stoppers(stopperTypes, stopping_measure):
        print(stopper)

    pass

"""