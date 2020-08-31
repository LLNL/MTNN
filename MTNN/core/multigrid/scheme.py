"""
Holds Multigrid Schemes
# TODO: Exception Handling/Checks
# TODO: Add logging level. Change Verbose option to logs level?
"""
# standard
from abc import ABC, abstractmethod

# third-party
import torch

#local
import MTNN.utils.logger as log
import MTNN.utils.printer as printer
import MTNN.utils.datatypes as mgdata

log = log.get_logger(__name__, write_to_file =True)

__all__ = ['Level',
           'Cascadic',
           'WCycle',
           'VCycle']


class Level:
    """A level or grid  in an Multigrid hierarchy"""
    def __init__(self, id: int, presmoother, postsmoother, prolongation, restriction,
                 coarsegrid_solver, corrector):
        """
        Args:
            id: <int> level id (assumed to be unique)
            model:  <core.components.model> Model
            presmoother:  <core.alg.multigrid.operators.smoother> Smoother
            postsmoother: <core.alg.multigrid.operators.smoother> Smoother
            prolongation: <core.alg.multigrid.operators.prolongation> Prolongation
            restriction: <core.alg.multigrid.operators.restriction> Restriction
            coarsegrid_solver:  <core.alg.multigrid.operators.smoother> Smoother
            corrector: <core.multigrid.operators.tau_corrector> TauCorrector
        """
        self.net = None
        self.id = id
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.coarsegrid_solver = coarsegrid_solver
        self.prolongation = prolongation
        self.restriction = restriction
        self.corrector = corrector

        # Data attributes
        # TODO: tokenize?
        self.interpolation_data = None
        # Lhs
        self.Winit = None  # list of tensors
        self.Binit = None  # list of tensors




    def presmooth(self, model, trainer):
        try:
            log.info(f'Applying presmoother {self.presmoother.__class__.__name__}')
            #self.set_rhs()
            self.presmoother.apply(model, trainer.dataloader, self.stopper, tau=self.corrector, verbose=trainer.verbose)
        except Exception:
            raise

    def postsmooth(self, model, trainer):
        try:
            log.info(f'Applying postsmoother {self.postsmoother.__class__.__name__}')
            #self.set_rhs()
            self.postsmoother.apply(model, trainer.dataloader, self.stopper, tau=self.corrector, verbose=trainer.verbose)
        except Exception:
           raise

    def coarse_solve(self, model, trainer):
        try:
            log.info(f'Applying coarse solve {self.coarsegrid_solver.__class__.__name__}')
            #self.set_rhs()
            self.coarsegrid_solver.apply(model, trainer.dataloader, self.stopper, tau=self.corrector,
                                         verbose=trainer.verbose)
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
            self.restriction.apply(fine_level, coarse_level, dataloader, self.corrector,  verbose)
        except Exception:
            raise

    def view(self):
        """Logs level attributes"""
        for atr in self.__dict__:

            atrval = self.__dict__[atr]
            if type(atrval) in (int, float, str, list, bool):
                log.info(f"\t{atr}: \t{atrval} ")
            elif isinstance(atrval, mgdata.operators):
                log.info(f"\t{atr}: \n\t\tRestriction: {atrval.R_op} "
                         f"\n\t\tProlongation: {atrval.P_op}")
            elif isinstance(atrval, mgdata.rhs):
                log.info(f"\t{atr}: {atrval}")
            else:
                log.info(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__}")



############################################################################
# Interface
############################################################################
class _BaseMultigridScheme(ABC):
    """
    Base Multigrid Hierarchy
    """
    def __init__(self, levels=None):
        """
        Args:
            levels: List of <core.alg.multigrid.multigrid.Level> Level objects
        """
        if levels is None:
            levels = []
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



class VCycle(_BaseMultigridScheme):
    def run(self, session, num_cycles: int):
        #TODO: Add checkpoints
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
            printer.print_level(self.levels)


        # Iteratively restrict each level's grid
        for cycle in range(num_cycles):
            #############################################
            # Down cycle - Coarsen/Restrict all levels
            ############################################
            for level_idx, level in enumerate(self.levels[:-1]):
                if session.trainer.verbose:
                    printer.print_levelstats(level_idx, num_levels, f"\nDOWN CYCLING {cycle=}: Restricting")

                fine_level = level
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # next level if it exists

                # Presmooth
                fine_level.presmooth(fine_level.net, session.trainer, session.trainer.verbose)

                #Restrict
                fine_level.restrict(fine_level, coarse_level, session.trainer.dataloader, session.trainer.verbose)

            # Smoothing with coarse-solver at the coarsest level
#            log.info(f"Scheme:Coarse-solving at the last level {self.levels[-1].net}")
            self.levels[-1].coarse_solve(level.net, session.trainer)

            ##############################################
            # Up Cycle - Interpolate/Prolongate back up to  all levels
            ##############################################
            for level_idx in range(num_levels - 2, -1, -1):
                if session.trainer.verbose:
                    printer.print_levelstats(level_idx, num_levels, "\nUP CYCLING: Prolongating")

                fine_level = self.levels[level_idx]
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # mod gets next level if it exists

                fine_level.prolong(fine_level, coarse_level, session.trainer.dataloader, session.trainer.verbose)
                fine_level.postsmooth(fine_level.net, session.trainer, session.trainer.verbose)

#                if session.trainer.verbose:
#                    printer.printLevelInfo(self.levels)

            # Return the fine net
            if session.trainer.verbose:
                log.info(f'================================='
                         f'Finished FAS Cycle'
                         f'=================================')
                printer.print_level(self.levels)

            return self.levels[0].net

