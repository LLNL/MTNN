"""
Holds Multigrid Schemes
# TODO: Exception Handling/Checks
# TODO: Add logging level. Change Verbose option to logs level?
"""
from abc import ABC, abstractmethod
import torch

#local
import MTNN.utils.logger as log
import MTNN.utils.printer as printer

log = log.get_logger(__name__, write_to_file =True)


class Level:
    """A level or grid  in an Multigrid hierarchy"""
    def __init__(self, presmoother=None, postsmoother=None,
                 prolongation=None, restriction=None,
                 coarsegrid_solver=None, stopping_measure=None):
        """
        Args:
            model: class <core.components.models.BaseModel>
            presmoother:  class <core.alg.multigrid.operators.smoother>
            postsmoother: class <core.alg.multigrid.operators.smoother>
            coarsener: # TODO
            prolongation: class <core.alg.multigrid.operators.prolongation>
            restriction: class <core.alg.multigrid.operators.restriction>
            coarsegrid_solver: class <core.alg.multigrid.operators.smoother>
            stopping: class <core.alg.multigrid.stopping>
        """
        self.net = None
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarsegrid_solver = coarsegrid_solver
        self.prolongation = prolongation
        self.restriction = restriction
        self.stopper = stopping_measure


    def presmooth(self, model, trainer):
        if hasattr(self, 'presmoother') and self.presmoother is not None:
            log.info(f'Applying presmoother {self.presmoother.__class__.__name__}')
            self.presmoother.apply(model, trainer.dataloader, self.stopper, trainer.verbose)
        else:
            pass

    def postsmooth(self, model, trainer):
        if hasattr(self, 'postsmoother') and self.postsmoother is not None:
            log.info(f'Applying postsmoother {self.postsmoother.__class__.__name__}')
            self.postsmoother.apply(model, trainer.dataloader, self.stopper, trainer.verbose)
        else:
            pass

    def coarse_solve(self, model, trainer):
        if hasattr(self, 'coarsegrid_solver') and self.coarsegrid_solver is not None:
            log.info(f'Applying coarse solve {self.coarsegrid_solver.__class__.__name__}')
            self.coarsegrid_solver.apply(model, trainer.dataloader, self.stopper, trainer.verbose)
        else:
            pass

    def prolong(self, fine_level, course_level, verbose):
        if hasattr(self, 'prolongation') and self.prolongation is not None:
            log.info(f'Applying prolongation {self.prolongation.__class__.__name__}')
            self.prolongation.apply(fine_level, course_level, verbose)
        else:
            pass

    def restrict(self, fine_level, course_level, verbose):
        if hasattr(self, 'restriction') and self.restriction is not None:
            log.info(f'Applying restriction {self.restriction.__class__.__name__}')
            self.restriction.apply(fine_level, course_level,verbose)
        else:
            pass

    def view(self):
        # TODO: print type
        for atr in self.__dict__:
            log.info(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__} ")



############################################################################
# Interface
############################################################################
class _BaseMultigridHierarchy(ABC):
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
class Cascadic(_BaseMultigridHierarchy):
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


class WCycle(_BaseMultigridHierarchy):
    pass
    # TODO

class FMG(_BaseMultigridHierarchy):
    pass
    # TODO


class VCycle(_BaseMultigridHierarchy):
    def run(self, session, num_cycles:int):
        """

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
            for i, level in enumerate(self.levels):
                log.info(f"Level {i}")
                level.view()


        # Iteratively restrict each level's grid
        for cycle in range(num_cycles):
            #############################################
            # Down cycle - Coarsen/Restrict all levels
            ############################################
            for level_idx, level in enumerate(self.levels):

                printer.printLevelStats(level_idx, num_levels, f"\nDown Cycle {cycle}: Coarsening ") #TODO: Remove

                fine_level = level
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)] # next level

                #fine_level.view() #TODO: Remove
                #coarse_level.view()
                printer.printModel(fine_level.net, msg = "Fine Level before Restriction", val = False, dim =True)
                printer.printModel(coarse_level.net, msg = "Coarse Level before Restriction", val = False, dim = True)
                #printer.printModel(fine_level.net, val=True, dim=True)

                # Presmooth
                fine_level.presmooth(fine_level.net, session.trainer)

                #Restrict
                fine_level.restrict(fine_level, coarse_level, session.trainer.verbose)
                #printer.printModel(fine_level.net, msg="Fine Level after Restriction", val=True, dim=True)
                #printer.printModel(coarse_level.net, msg = "Coarse Level after Restriction", val = True, dim = True)

            # Smoothing with coarse-solver at the coarsest level
            log.info(f"Scheme:Coarse-solving at the last level {self.levels[-1].net}")
            self.levels[-1].coarse_solve(level.net, session.trainer)

            ##############################################
            # Up Cycle - Prolongate back up to  all levels
            ##############################################
            for level_idx in range(num_levels - 2, -1, -1):

                printer.printLevelStats(level_idx, num_levels, "\nUp Cycle: Prolongating")

                fine_grid = self.levels[level_idx]

                #coarse_level = self.levels[lev_id + 1]
                fine_grid.prolong(fine_level, coarse_level, session.trainer.verbose)
                fine_grid.postsmooth(fine_grid.net, session.trainer)


            log.info(f" \nFinished FAS Cycle")
            # Return the fine net
            return self.levels[0].net

