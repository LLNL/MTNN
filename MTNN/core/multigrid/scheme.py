"""
Holds Multigrid Schemes
# TODO: Exception Handling/Checks
# TODO: Add logging level. Change Verbose option to logs level?
"""
# standard
from collections import namedtuple
from abc import ABC, abstractmethod


# third-party
import torch

#local
import MTNN.utils.logger as log
import MTNN.utils.printer as printer

log = log.get_logger(__name__, write_to_file =True)


class Level:
    """A level or grid  in an Multigrid hierarchy"""
    def __init__(self, id: int, presmoother, postsmoother, prolongation, restriction,
                 coarsegrid_solver, stopping_measure, loss_fn):
        """
        Args:
            id: <int> level id (assumed to be unique)
            model:  <core.components.models.BaseModel>
            presmoother:  <core.alg.multigrid.operators.smoother>
            postsmoother: <core.alg.multigrid.operators.smoother>
            prolongation: <core.alg.multigrid.operators.prolongation>
            restriction: <core.alg.multigrid.operators.restriction>
            coarsegrid_solver:  <core.alg.multigrid.operators.smoother>
            stopping:  <core.alg.multigrid.stopping>
        """
        self.net = None
        self.id = id
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarsegrid_solver = coarsegrid_solver
        self.prolongation = prolongation
        self.restriction = restriction
        self.stopper = stopping_measure
        self.loss_fn = loss_fn

        # Computation attributes
        # TODO: pipeline and tokenize?
        self.interpolation_data = None
        # Lhs
        self.Winit_array = None
        self.Binit_array = None
        # Rhs
        self.rhs_W_array = None
        self.rhs_B_array = None

        RHS = namedtuple('rhs', ['W', 'b'])
        self.rhs = RHS(self.rhs_W_array, self.rhs_B_array)

    def presmooth(self, model, trainer):
        try:
            log.info(f'Applying presmoother {self.presmoother.__class__.__name__}')
            self.presmoother.apply(model, trainer.dataloader, self.stopper, self.rhs,
                                trainer.verbose)
        except Exception:
            raise

    def postsmooth(self, model, trainer):
        try:
            log.info(f'Applying postsmoother {self.postsmoother.__class__.__name__}')
            self.postsmoother.apply(model, trainer.dataloader, self.stopper, self.rhs,
                                    trainer.verbose)
        except Exception:
           raise

    def coarse_solve(self, model, trainer):
        try:
            log.info(f'Applying coarse solve {self.coarsegrid_solver.__class__.__name__}')
            self.coarsegrid_solver.apply(model, trainer.dataloader, self.stopper, self.rhs, trainer.verbose)
        except Exception:
            raise

    def prolong(self, fine_level, coarse_level, dataloader, verbose):
        try:
            log.info(f'Applying prolongation {self.prolongation.__class__.__name__}')
            self.prolongation.apply(fine_level, coarse_level, dataloader, verbose)
        except Exception:
            raise

    def restrict(self, fine_level, coarse_level, dataloader, verbose):
        try:
            log.info(f'Applying restriction {self.restriction.__class__.__name__}')
            self.restriction.apply(fine_level, coarse_level, dataloader,  verbose)
        except Exception:
            raise

    def view(self):
        # TODO: Add modality
        for atr in self.__dict__:
            if type(self.__dict__[atr]) in (int, float, str, list, bool):
                log.info(f"\t{atr}: \t{self.__dict__[atr]} ")
            else:
                log.info(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__}")


############################################################################
# Interface
############################################################################
class _BaseMultigridScheme(ABC):
    """
    Base Multigrid Hierarchy
    """
    def __init__(self, levels=[]):
        """
        Args:
            levels: List of <core.alg.multigrid.multigrid.Level> Level objects
        """
        self.levels = levels


    @abstractmethod
    def run(self, **kawrgs):
        raise NotImplementedError

    def __len__(self):
        return len(self.levels)


############################################################################
# Implementations
############################################################################
class Cascadic(_BaseMultigridScheme):
    """
    Interface for Cascadic Multigrid Algorithm
    """
    def run(self, session):
        """
        Args:
            session: Model, Trainer
        Returns:

        """

        # Verbose
        if session.trainer.verbose:
            log.info(f" Applying Cascadic Multigrid")
            log.info(f"\nNumber  of levels: {self.get_num_levels()}")
            for i, level in enumerate(self.levels):
                log.info(f"Level {i}")
                level.view()

        # Loading
        if session.trainer.load:
            log.info(f"\nLoading from {session.trainer.load_path}")
            model = torch.load(session.trainer.load_path)
            model.eval()

        # Training
        for level_idx, level in enumerate(self.levels):
            log.info(f"\nLevel  {level_idx}: Applying Presmoother ")
            level.presmooth(session.model, session.trainer)

            log.info(f"\nLevel {level_idx} :Applying Prolongation")
            level.prolong(session)

            log.info(f"\nLevel {level_idx}: Appying Coarse Solver")
            level.coarse_solve(session)

            # Apply last layer smoothing

            if level_idx == self.levels[-1]:
                log.info(f"\nLevel {level_idx}: Appying Postsmoother")
                level.postsmooth(session)

        # Saving
        if session.trainer.save:
            log.info(f"\nSaving to ...{session.trainer.save_path}")
            torch.save({'model_state_dict': model.state_dict()},
                       session.trainer.save_path)


class WCycle(_BaseMultigridScheme):
    pass
    # TODO

class VCYCLE(_BaseMultigridScheme):
    pass
    # TODO


class FASVCycle(_BaseMultigridScheme):
    def run(self, session, num_cycles:int):
        """
        Full Approximation Scheme Multigrid Algorithm

        Attributes:
            levels: <list>

        Args:
            session: <MTNN.trainer.Session> holds original model, trainer settings
            num_cycles: <int> Number of iterations

        Returns:

        """
        # Initiate first level
        num_levels = len(self.levels)
        self.levels[0].net = session.model

        if session.trainer.verbose:
            log.info(f"Applying FAS cycle with {num_levels} levels")
            printer.printLevelInfo(self.levels)


        # Iteratively restrict each level's grid
        for cycle in range(num_cycles):
            #############################################
            # Down cycle - Coarsen/Restrict all levels
            ############################################
            for level_idx, level in enumerate(self.levels[:-1]):
                if session.trainer.verbose:
                    printer.printLevelStats(level_idx, num_levels, f"\nDOWN CYCLING {cycle}: Restricting") #TODO: Remove

                fine_level = level
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # next level if it exists

                # Presmooth
                fine_level.presmooth(fine_level.net, session.trainer)

                #Restrict
                fine_level.restrict(fine_level, coarse_level, session.trainer.dataloader, session.trainer.verbose)

                log.debug(f"Down Cycling {cycle = } {level_idx = }")
                printer.printLevelInfo(self.levels)

            # Smoothing with coarse-solver at the coarsest level
            log.info(f"Scheme:Coarse-solving at the last level {self.levels[-1].net}")
            self.levels[-1].coarse_solve(level.net, session.trainer)

            ##############################################
            # Up Cycle - Interpolate/Prolongate back up to  all levels
            ##############################################
            for level_idx in range(num_levels - 2, -1, -1):
                if session.trainer.verbose: #TODO: Remove
                    printer.printLevelStats(level_idx, num_levels, "\nUP CYCLING: Prolongating")

                fine_level = self.levels[level_idx]
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # next level if it exists

                fine_level.prolong(fine_level, coarse_level, session.trainer.dataloader, session.trainer.verbose)
                fine_level.postsmooth(fine_level.net, session.trainer)

                if session.trainer.verbose:
                    printer.printLevelInfo(self.levels)


            # Return the fine net
            if session.trainer.verbose:
                log.info(f" \nFinished FAS Cycle")
                printer.printLevelInfo(self.levels)

            return self.levels[0].net

