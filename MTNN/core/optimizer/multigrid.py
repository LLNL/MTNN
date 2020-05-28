"""
Holds Multigrid Schemes
# TODO: Exception Handling/Checks
# TODO: Add logging level. Change Verbose option to logs level?
"""
import torch
from abc import ABC, abstractmethod

#local
import MTNN.utils.logger as log

log = log.get_logger(__name__, write_to_file =True)

class Level:
    """A level in an Multigrid hierarchy"""
    def __init__(self, presmoother=None, postsmoother=None,
                 prolongation=None, restriction=None,
                 coarsegrid_solver=None, stopping_criteria=None):
        """
        Args:
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
            log.info(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__} ")

##############################
# API
##############################
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
    def run(self, **kawrgs):
        raise NotImplementedError

    def get_num_levels(self):
        return len(self.levels)

############################################################################
# Implementations
############################################################################
class Cascadic(_BaseMultigridHierarchy):
    """
    Interface for Cascadic Multigrid Algorithm
    """
    def run(self, model, trainer):

        # Verbose
        if trainer.verbose:
            log.info(f"\nNumber  of levels: {self.get_num_levels()}")
            for i, level in enumerate(self.levels):
                log.info(f"Level {i}")
                level.view()

        # Loading
        if trainer.load:
            log.info(f"\nLoading from {trainer.load_path}")
            model = torch.load(trainer.load_path)
            model.eval()

        # Training
        for level_idx, level in enumerate(self.levels):
            log.info(f"\nLevel  {level_idx}: Applying Presmoother ")
            level.presmooth(model, trainer.dataloader, trainer.verbose)

            log.info(f"\nLevel {level_idx} :Applying Prolongation")
            level.prolong(model, trainer.verbose)

            log.info(f"\nLevel {level_idx}: Appying Coarse Solver")
            level.coarse_solve(model, trainer.dataloader, trainer.verbose)

            # Apply last layer smoothing

            if level_idx == self.levels[-1]:
                log.info(f"\nLevel {level_idx}: Appying Postsmoother")
                level.postsmooth(model, trainer.dataloader, trainer.verbose)

        # Saving
        if trainer.save:
            log.info(f"\nSaving to ...{trainer.save_path}")
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

