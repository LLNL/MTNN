"""
Holds Multigrid Schemes
# TODO: Exception Handling/Checks
# TODO: Add logging level. Change Verbose option to logs level?
"""
import torch
from abc import ABC, abstractmethod


class Level:
    """A level in an Multigrid hierarchy"""
    def __init__(self, presmoother=None, postsmoother=None,
                 prolongation=None, restriction=None,
                 coarsegrid_solver=None, stopping_criteria=None):
        """
        Args:
            num_epochs: <int>
            presmoother: <core.alg.optimizer.operators.smoother>
            postsmoother: <core.alg.optimizer.operators.smoother>
            prolongation: <core.alg.optimizer.operators.prolongation>
            restriction: <core.alg.optimizer.operators.restriction>
            coarsegrid_solver: <core.alg.optimizer.operators.smoother>
            stopping_criteria: <core.alg.optimizer.stopping>
        """
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarsegrid_solver = coarsegrid_solver
        self.prolongation = prolongation
        self.restrictrion = restriction
        self.stopper = stopping_criteria

    def presmooth(self, model, dataloader, verbose):
        assert(hasattr(self, 'presmoother'))
        self.presmoother.apply(model, dataloader, self.stopper, verbose)

    def postsmooth(self, model, dataloader, verbose):
        assert(hasattr(self, 'postsmoother'))
        self.postsmoother.apply(model, dataloader, self.stopper,verbose)

    def coarse_solve(self, model, dataloader, verbose):
        assert(hasattr(self, 'coarsegrid_solver'))
        self.coarsegrid_solver.apply(model, dataloader, self.stopper, verbose)

    def prolong(self, model, verbose):
        assert(hasattr(self, 'prolongation'))
        self.prolongation.apply(model, verbose)

    def restrict(self, model, verbose):
        assert(hasattr(self, 'restriction'))
        self.restriction.apply(model, verbose)

    def view(self):
        # TODO: print type
        for atr in self.__dict__:
            print(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__} ")


class _BaseMultigridHierarchy(ABC):
    """
    Base Multigrid Hierarchy
    """
    def __init__(self, levels=[]):
        """

        Args:
            levels:

        """
        self.levels = levels

    @abstractmethod
    def run(self):
        raise NotImplementedError

    def get_num_levels(self):
        return len(self.levels)


class Cascadic(_BaseMultigridHierarchy):
    """
    Interface for Cascadic Multigrid Algorithm
    """
    #def run(self, model, data, verbose:bool, save=False, path="./model"):
    def run(self, model, trainer):

        # Verbose
        if trainer.verbose:
            print(f"\nNumber  of levels: {self.get_num_levels()}")
            for i, level in enumerate(self.levels):
                print(f"Level {i}")
                level.view()

        # Loading
        if trainer.load:
            print(f"\nLoading from {trainer.load_path}")
            model = torch.load(trainer.load_path)
            model.eval()

        # Training
        """"
        for level_idx, level in enumerate(self.levels):
            # Coarse Solve
            if level_idx == 0:
                print(f"Level {level_idx}: Appying Coarse Solver")
                level.coarse_solve(model, trainer.dataloader, trainer.verbose)

            # Pre-Smooth
            elif level_idx == (len(self.levels) - 1):
                print(f"Level  {level_idx}: Applying Presmoother ")
                level.presmooth(model, trainer.dataloader, trainer.verbose)

            # Prolongation/Interpolation
            else:
                print(f"Level {level_idx} :Applying Prolongation")
                level.prolong(model, trainer.verbose)

        """

        for level_idx in range(len(self.levels)):
            print(f"\nLevel  {level_idx}: Applying Presmoother ")
            level.presmooth(model, trainer.dataloader, trainer.verbose)

            print(f"\nLevel {level_idx} :Applying Prolongation")
            level.prolong(model, trainer.verbose)

            print(f"\nLevel {level_idx}: Appying Coarse Solver")
            level.coarse_solve(model, trainer.dataloader, trainer.verbose)

            # Apply last layer smoothing
            if level == self.levels[-1]:
                print(f"\nLevel {level_idx}: Appying Postsmoother")
                level.postsmooth(model, trainer.dataloader, trainer.verbose)

        # Saving
        if trainer.save:
            print(f"\nSaving to ...{trainer.save_path}")
            torch.save({'model_state_dict': model.state_dict()},
                       trainer.save_path)



class VCycle(_BaseMultigridHierarchy):
    pass
    # TODO

class WCycle(_BaseMultigridHierarchy):
    pass
    # TODO

class FMG(_BaseMultigridHierarchy):
    pass
    # TODO

class FAS(_BaseMultigridHierarchy):
    pass
    # TODO

