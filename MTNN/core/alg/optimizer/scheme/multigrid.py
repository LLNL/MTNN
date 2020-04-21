"""
Holds Multigrid Schemes
# TODO: Exception Handling/ Checks
"""
import MTNN.core.alg.optimizer.operators.smoother as smoother
import MTNN.core.alg.optimizer.operators.prolongation as prolongation


class Level:
    """A level in an MG hierarchy"""
    def __init__(self, presmoother=None, postsmoother=None, prolongation=None, refinement=None, coarse_solver=None):
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.prolongation = prolongation
        self.refinement = refinement
        self.coarse_solver = coarse_solver

class BaseMultigrid():
    """
    Base Multigrid Attributes
    """
    def __init__(self, levels=0, presmoother=None , postsmoother=None, finegrid_solver=None, coarsegrid_solver=None,
                 prolongation=None, restriction=None, stopping_criteria=None):
        """

        Args:
            levels:
            presmoother: <smoother>
            postsmoother: <smoother>
            coarsegrid_solver:
            prolongation: <prolongation>
            restriction: <restriction>
            stopping_criteria:
        """

        self.levels = levels
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarsegrid_solver = coarsegrid_solver
        self.finegrid_solver = finegrid_solver
        self.prolongation = prolongation
        self.restriction = restriction
        self.stopping_criteria = stopping_criteria

    def presmooth(self, model):
        raise NotImplementedError

    def restrict(self, model):
        raise NotImplementedError

    def coarse_solve(self, model, data, stopping):
        raise NotImplementedError

    def prolong(self, model):
        raise NotImplementedError

    def fineg_solve(self, model):
        raise NotImplementedError

    def postsmooth(self, model, data, stopping):
        raise NotImplementedError


class Cascadic(BaseMultigrid):
    """
    Interface for Cascadic Multigrid Algorithm
    """

    def presmooth(self, model):
        pass

    def restrict(self, model):
        pass

    def coarse_solve(self, model, data, stopping):
        # Apply SGD until stopping criteria
        self.coarsegrid_solver.apply(model, data, stopping)
        return model

    def prolong(self, model):
        self.prolongation.apply(model)
        return model

    def fine_solve(self, model, data, stopping):
        self.finegrid_solver.apply(model, data, stopping)
        return model

    def postsmooth(self, model, data, stopping):
        self.postsmoother.apply(model, data, stopping)
        return model

    def run(self, model, data):
         # Apply to each level
        for level in range(self.levels):
            model = self.coarse_solve(model, data, self.stopping_criteria)
            model = self.prolong(model)
            model = self.postsmooth(model, data, self.stopping_criteria)

            # Save and Checkpoint
            # TODO

        return model




class VCycle(BaseMultigrid):
    pass
    # TODO

class WCycle(BaseMultigrid):
    pass
    # TODO

class FMG(BaseMultigrid):
    pass
    # TODO

class FAS(BaseMultigrid):
    pass
    # TODO

