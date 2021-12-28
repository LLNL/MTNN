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
from MTNN.utils import deviceloader

log = log.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['Cascadic',
           'WCycle',
           'VCycle',
           'FMG']

############################################################################
# Interface
############################################################################
class BaseMultigridScheme(ABC):
    """
    Base Multigrid Hierarchy
    """
    def __init__(self, levels, cycles, subsetloader, validation_callback):
        """@param levels <List[Level]>

        @param cycles <int> Number of cycle iterations

        @param subsetloader
        <core.alg.multigrid.operators.subsetloader> Create a new
        dataloader focused on a subset of data for each cycle.

        @param validation_callback <ValidationCallback> A function to
        call after every training cycle to measure performance.

        """
        if levels is None:
            levels = []
        self.levels = levels
        self.num_levels = len(self.levels)
        self.cycles = cycles
        self.subsetloader = subsetloader
        self.validation_callback = validation_callback

    def setup(self, model):
        """Set the first level's model"""
        self.levels[0].net = model

    @abstractmethod
    def run(self, model, dataloader, trainer):
        raise NotImplementedError

    def __len__(self):
        return len(self.levels)


############################################################################
# Concrete cycles
############################################################################
class VCycle(BaseMultigridScheme):
    def run(self, dataloader, verbose):
        """
        V-Cycle Multilevel Training method

        @param dataloader
        @param verbose

        """
        if verbose:
            printer.print_cycleheader(self)
            printer.print_level(self.levels)

        # Iteratively restrict each level's grid
        for cycle in range(0, self.cycles):
            if verbose:
                printer.print_cycle_status(self, cycle)
            #############################################
            # Down cycle - Coarsen/Restrict all levels
            ############################################

            for level_ind in range(self.num_levels-1):
                if verbose:
                    printer.print_levelstats(cycle, self.cycles, level_ind, self.num_levels, f"DOWN CYCLING ")
                cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)

                fine_level = self.levels[level_ind]
                coarse_level = self.levels[level_ind + 1]

                # Presmooth
                fine_level.presmooth(fine_level.net, cycle_dataloader, verbose)

                # Restrict
                fine_level.restrict(fine_level, coarse_level, cycle_dataloader, verbose)

            # Smoothing with coarse-solver at the coarsest level
            cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
            self.levels[self.num_levels-1].coarse_solve(self.levels[self.num_levels-1].net, cycle_dataloader, verbose)

            ##############################################
            # Up Cycle - Interpolate/Prolongate back up to  all levels
            ##############################################
            for level_ind in range(self.num_levels - 2, -1, -1):
                if verbose:
                    printer.print_levelstats(cycle, self.cycles, level_ind, self.num_levels, f"\nUP CYCLING")
                cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)

                fine_level = self.levels[level_ind]
                coarse_level = self.levels[(level_ind + 1) % len(self.levels)]  # mod gets next level if it exists

                fine_level.prolong(fine_level, coarse_level, cycle_dataloader, verbose)
                fine_level.postsmooth(fine_level.net, cycle_dataloader, verbose)


            if self.validation_callback is not None:
                self.validation_callback(self.levels, cycle)

class WCycle(BaseMultigridScheme):
    def iterate_on_level(self, dataloader, level_ind, verbose):
        if level_ind == self.num_levels-1:
            # on coarsest level
            cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
            self.levels[level_ind].coarse_solve(
                self.levels[level_ind].net, cycle_dataloader, verbose)
            return

        # presmoothing
        cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
        fine_level = self.levels[level_ind]
        coarse_level = self.levels[level_ind+1]
        fine_level.presmooth(fine_level.net, cycle_dataloader, verbose)

        # go to coarse level twice because it's a W cycle
        fine_level.restrict(fine_level, coarse_level, cycle_dataloader, verbose)
        self.iterate_on_level(dataloader, level_ind+1, verbose)
        fine_level.prolong(fine_level, coarse_level, cycle_dataloader, verbose)
        fine_level.restrict(fine_level, coarse_level, cycle_dataloader, verbose)
        self.iterate_on_level(dataloader, level_ind+1, verbose)

        # postsmoothing
        cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
        fine_level = self.levels[level_ind]
        coarse_level = self.levels[level_ind + 1]
        fine_level.prolong(fine_level, coarse_level, cycle_dataloader, verbose)
        fine_level.postsmooth(fine_level.net, cycle_dataloader, verbose)

    def run(self, dataloader, verbose):
        """W-cycle training method

        @param dataloader
        @param verbose

        """
        if verbose:
            printer.print_cycleheader(self)
            printer.print_level(self.levels)

        # Iteratively restrict each level's grid
        for cycle in range(0, self.cycles):
            if verbose:
                printer.print_cycle_status(self, cycle)
                
            self.iterate_on_level(dataloader, 0, verbose)
            
            if self.validation_callback is not None:
                self.validation_callback(self.levels, cycle)


def FMG(BaseMultigridScheme):
    pass
    # TODO
