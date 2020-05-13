import MTNN.core.optimizer.multigrid as mg


def build_uniform(num_levels=0, presmoother=None, postsmoother=None,
                  prolongation=None, restriction=None, coarsegrid_solver=None, stopping_criteria=None):
    """
    Constructs a uniform set of levels to pass to BaseMultigrid.
    Args:
        num_levels:
        presmoother:
        postsmoother:
        prolongation:
        restriction:
        coarsegrid_solver:
        stopping_criteria:

    Returns:
        set_of_levels: <list> of Level Objects

    """
    set_of_levels = []

    for num in range(num_levels):
        aLevel = mg.Level(presmoother, postsmoother, prolongation,
                          restriction, coarsegrid_solver, stopping_criteria)
        set_of_levels.append(aLevel)
    return set_of_levels


