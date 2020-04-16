"""
Holds Multigrid Schemes
"""

class BaseMultigrid():
    """
    Base Multigrid Attributes
    """
    def __init__(self, levels=0, presmoother=None, postsmoother=None, coarsegrid_solver=None, prolongation=None, restriction=None, stopping_criteria=None):
        self.levels = levels
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarse_grid_solver = coarsegrid_solver
        self.prolongation = prolongation
        self.restriction = restriction
        self.stopping_criteria = stopping_criteria


class Cascadic(BaseMultigrid):
    ...



class VCycle(BaseMultigrid):

    def __init__(self):
        pass

class WCycle(BaseMultigrid):

    def __init__(self):
        pass

class FMG(BaseMultigrid):

    def __init__(self):
        pass