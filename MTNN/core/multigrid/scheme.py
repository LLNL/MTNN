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
from MTNN.core.components.subsetloader import WholeSetLoader
from MTNN.utils import deviceloader

log = log.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['Level',
           'Cascadic',
           'WCycle',
           'VCycle',
           'FMG']


class Level:
    """A level or grid  in an Multigrid hierarchy"""
    def __init__(self, id: int, presmoother, postsmoother, prolongation, restriction,
                 coarsegrid_solver, num_epochs, corrector=None):
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
        self.num_epochs =  num_epochs
        self.l2_info = None

        # Data attributes
        # TODO: tokenize?
        self.interpolation_data = None
        # Lhs
        self.Winit = None
        self.Binit = None

    def presmooth(self, model, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'PRESMOOTHING {self.presmoother.__class__.__name__}',))
            self.presmoother.apply(model, dataloader, self.num_epochs, tau=self.corrector,
                                   l2_info = self.l2_info, verbose=verbose)
        except Exception:
            raise

    def postsmooth(self, model, dataloader , verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'POSTSMOOTHING {self.postsmoother.__class__.__name__}'))
            self.postsmoother.apply(model, dataloader, self.num_epochs, tau=self.corrector,
                                    l2_info = self.l2_info, verbose=verbose)
        except Exception:
           raise

    def coarse_solve(self, model, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'COARSE SOLVING {self.coarsegrid_solver.__class__.__name__}', border="*"))
            self.coarsegrid_solver.apply(model, dataloader, self.num_epochs, tau=self.corrector,
                                         l2_info = self.l2_info,verbose=verbose)
        except Exception:
            raise

    def prolong(self, fine_level, coarse_level, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'PROLONGATING {self.prolongation.__class__.__name__}'))

            self.prolongation.apply(fine_level, coarse_level, dataloader, verbose)
        except Exception:
            raise

    def restrict(self, fine_level, coarse_level, dataloader, verbose=False):
        try:
            if verbose:
                log.info(printer.format_header(f'RESTRICTING {self.restriction.__class__.__name__}'))
            self.restriction.apply(fine_level, coarse_level,  dataloader,  verbose)
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
    def __init__(self, levels=None, cycles=1, subsetloader = WholeSetLoader(),
                 depth_selector = None):
        """
        Args:
            levels: List of <core.alg.multigrid.multigrid.Level> Level objects
            cycles: <int> Number of cycle iterations
            subsetloader: <core.alg.multigrid.operators.subsetloader> Create a
                           new dataloader focused on a subset of data for each 
                           cycle.
            depth_selector: A function that takes the cycle index as input and 
                            returns the hierarchy depth for this cycle.
        """
        if levels is None:
            levels = []
        self.levels = levels
        self.cycles = cycles
        self.subsetloader = subsetloader
        if depth_selector is None:
            self.depth_selector = lambda c : len(self.levels)
        else:
            self.depth_selector = depth_selector

    def setup(self, model):
        """Set the first level's model"""
        self.levels[0].net = model

    @abstractmethod
    def run(self, model, dataloader, trainer):
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
    def run(self, model, dataloader, trainer):
        """
        Args:
            model: <MTNN.core.components.model> Model
            trainer: <MTNN.core.alg.trainer> Trainer
            num_cycles: <int> Number of cycle iterations
        Returns:


        """

        # Verbose
        if trainer.verbose:
            log.info(f" Applying Cascadic Multigrid")
            log.info(f"\nNumber  of levels: {len(self.levels)}")
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

            # Set levels
            fine_level = level
            coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # next level if it exists

            log.info(f"\nLevel  {level_idx}: Applying Presmoother ")
            level.presmooth(model, trainer)

            log.info(f"\nLevel {level_idx} :Applying Prolongation")
            level.prolong(coarse_level, dataloader, trainer.verbose)

            log.info(f"\nLevel {level_idx}: Appying Coarse Solver")
            level.coarse_solve(level.net, trainer)

            # Apply last layer smoothing
            if level_idx == self.levels[-1]:
                log.info(f"\nLevel {level_idx}: Appying Postsmoother")
                level.postsmooth(model, trainer)

        # Saving
        if trainer.save:
            log.info(f"\nSaving to ...{trainer.save_path}")
            torch.save({'model_state_dict': model.state_dict()},
                       trainer.save_path)

        return self.levels[-1].net


class WCycle(_BaseMultigridScheme):
    pass
    # TODO


class VCycle(_BaseMultigridScheme):
    def run(self, model, dataloader, trainer):
        #TODO: Add checkpoints
        """
        Basic V-Cycle Scheme

        Args:
            model: <MTNN.core.components.models> subclass of BaseModel
            dataloader: <MTNN.core.components.data> subclass of BaseDataLoader

        Returns:

        """
        # Initiate first level
        self.levels[0].net = model

        if trainer.verbose:
            printer.print_cycleheader(self)
            printer.print_level(self.levels)

        self.best_seen = [10000] * len(self.levels)
        
        # Iteratively restrict each level's grid
        for cycle in range(0, self.cycles):
            # if cycle == 2000:
            #     for level in self.levels:
            #         level.presmoother.reduce_lr(0.5)
            if trainer.verbose:
                printer.print_cycle_status(self, cycle)
            #############################################
            # Down cycle - Coarsen/Restrict all levels
            ############################################

            num_levels = self.depth_selector(cycle)
            for level_idx in range(num_levels-1):
                if trainer.verbose:
                    printer.print_levelstats(cycle, self.cycles, level_idx, num_levels, f"DOWN CYCLING ")
                cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)

                fine_level = self.levels[level_idx]
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # next level if it exists

                # Presmooth
                fine_level.presmooth(fine_level.net, cycle_dataloader, trainer.verbose)

                # Restrict
                fine_level.restrict(fine_level, coarse_level, cycle_dataloader, trainer.verbose)

            # Smoothing with coarse-solver at the coarsest level
            cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
            self.levels[num_levels-1].coarse_solve(self.levels[num_levels-1].net, cycle_dataloader, trainer.verbose)

            ##############################################
            # Up Cycle - Interpolate/Prolongate back up to  all levels
            ##############################################
            for level_idx in range(num_levels - 2, -1, -1):
                if trainer.verbose:
                    printer.print_levelstats(cycle, self.cycles, level_idx, num_levels, f"\nUP CYCLING")
                cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)

                fine_level = self.levels[level_idx]
                coarse_level = self.levels[(level_idx + 1) % len(self.levels)]  # mod gets next level if it exists

                fine_level.prolong(fine_level, coarse_level, cycle_dataloader, trainer.verbose)
                fine_level.postsmooth(fine_level.net, cycle_dataloader, trainer.verbose)


            if trainer.verbose and (cycle + 1) % 5 == 0:
                for level in self.levels:
                    level.net.eval()
                # with torch.no_grad():
                #     total_loss = [0.0] * len(self.levels)
                #     for mini_batch_data in dataloader:
                #         inputs, true_outputs  = deviceloader.load_data(mini_batch_data, self.levels[0].net.device)
                #         for level_ind, level in enumerate(self.levels):
                #             outputs = level.net(inputs)
                #             total_loss[level_ind] += level.presmoother.loss_fn(outputs, true_outputs)
                #     for level_ind in range(len(self.levels)):
                #         print("Level {}: After {} cycles, training loss is {}".format(level_ind, cycle, total_loss[level_ind]), flush=True)
                with torch.no_grad():
                    total_test_loss = [0.0] * len(self.levels)
                    test_linf_loss = [0.0] * len(self.levels)
                    for mini_batch_data in self.test_loader:
                        inputs, true_outputs  = deviceloader.load_data(mini_batch_data, self.levels[0].net.device)
                        for level_ind, level in enumerate(self.levels):
                            outputs = level.net(inputs)
                            total_test_loss[level_ind] += level.presmoother.loss_fn(outputs, true_outputs)
                            linf_temp = torch.max (torch.max(torch.abs(true_outputs - outputs), dim=1).values)
                            test_linf_loss[level_ind] = max(linf_temp, test_linf_loss[level_ind])
                    for level_ind in range(len(self.levels)):
                        if total_test_loss[level_ind] < self.best_seen[level_ind]:
                            self.best_seen[level_ind] = total_test_loss[level_ind]
#                        print("Level {}: After {} cycles, validation loss is {}, best seen is {}, max err is {}".format(level_ind, cycle, total_test_loss[level_ind], self.best_seen[level_ind], test_linf_loss[level_ind]), flush=True)
                        print("Level {}: After {} cycles, validation loss is {}, best seen is {}, linf loss is {}".format(level_ind, cycle, total_test_loss[level_ind], self.best_seen[level_ind], test_linf_loss[level_ind]), flush=True)
                for level in self.levels:
                    level.net.train()

        return self.levels[0].net

def FMG(_BaseMultigridScheme):
    pass
    # TODO
