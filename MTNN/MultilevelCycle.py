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
from MTNN.utils import logger

# Public
__all__ = ['WCycle',
           'VCycle',
           'FMG']

############################################################################
# Interface
############################################################################
class MultilevelCycle(ABC):
    """
    Base Multigrid Hierarchy
    """
    def __init__(self, levels, cycles, subsetloader, validation_callbacks):
        """@param levels <List[Level]>

        @param cycles <int> Number of cycle iterations

        @param subsetloader
        <core.alg.multigrid.operators.subsetloader> Create a new
        dataloader focused on a subset of data for each cycle.

        @param validation_callback <List(ValidationCallback)> A list of functions to
        call after every training cycle to measure performance.

        """
        if levels is None:
            levels = []
        self.levels = levels
        self.num_levels = len(self.levels)
        self.cycles = cycles
        self.subsetloader = subsetloader
        self.validation_callbacks = validation_callbacks
        self.log = logger.get_MTNN_logger()

    def setup(self, model):
        """Set the first level's model"""
        self.levels[0].net = model

    @abstractmethod
    def run(self, model, dataloader, trainer):
        raise NotImplementedError

    def log_cycling_header(self):
        self.log.warning("Applying {}".format(self.__class__.__name__).center(100, '='))
        for ind, level in enumerate(self.levels):
            self.log.warning("Level {}".format(ind))
            level.view()
    
    def __len__(self):
        return len(self.levels)


############################################################################
# Concrete cycles
############################################################################
class VCycle(MultilevelCycle):
    def log_cycle_status(self, cycle, num_cycles):
        self.log.info("\n" + f"CYCLE {cycle+1} / {num_cycles}".center(100, "="))
    def log_level_status(self, cycle, max_cycles, level_ind, num_levels, msg):
        self.log.info(f"{msg} Cycle {cycle+1}/{max_cycles} Level {level_ind+1}/{num_levels}")

    
    def run(self, dataloader):
        """
        V-Cycle Multilevel Training method

        @param dataloader
        @param verbose

        """
        self.log_cycling_header()

        # Iteratively restrict each level's grid
        for cycle in range(0, self.cycles):
            self.log_cycle_status(cycle, self.cycles)
                
            #############################################
            # Down cycle - Coarsen/Restrict all levels
            ############################################
            for level_ind in range(self.num_levels-1):
                self.log_level_status(cycle, self.cycles, level_ind, self.num_levels, "DOWN CYCLING")
                cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)

                fine_level = self.levels[level_ind]
                coarse_level = self.levels[level_ind + 1]

                # Presmooth
                fine_level.presmooth(fine_level.net, cycle_dataloader)

                # Restrict
                fine_level.restrict(fine_level, coarse_level, cycle_dataloader)

            #############################################
            # Coarsest level
            ############################################
            
            # Smoothing with coarse-solver at the coarsest level
            cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
            self.log_level_status(cycle, self.cycles, self.num_levels-1, self.num_levels, "COARSE SMOOTHING")
            self.levels[self.num_levels-1].coarse_solve(self.levels[self.num_levels-1].net, cycle_dataloader)

            ##############################################
            # Up Cycle - Interpolate/Prolongate back up to  all levels
            ##############################################
            for level_ind in range(self.num_levels - 2, -1, -1):
                self.log_level_status(cycle, self.cycles, level_ind, self.num_levels, "UP CYCLING")
                cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)

                fine_level = self.levels[level_ind]
                coarse_level = self.levels[(level_ind + 1) % len(self.levels)]  # mod gets next level if it exists

                fine_level.prolong(fine_level, coarse_level, cycle_dataloader)
                fine_level.postsmooth(fine_level.net, cycle_dataloader)


            if self.validation_callbacks is not None:
                for vc in self.validation_callbacks:
                    vc(self.levels, cycle)

class WCycle(MultilevelCycle):
    def iterate_on_level(self, dataloader, level_ind):
        if level_ind == self.num_levels-1:
            # on coarsest level
            cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
            self.levels[level_ind].coarse_solve(
                self.levels[level_ind].net, cycle_dataloader)
            return

        # presmoothing
        cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
        fine_level = self.levels[level_ind]
        coarse_level = self.levels[level_ind+1]
        fine_level.presmooth(fine_level.net, cycle_dataloader)

        # go to coarse level twice because it's a W cycle
        fine_level.restrict(fine_level, coarse_level, cycle_dataloader)
        self.iterate_on_level(dataloader, level_ind+1)
        fine_level.prolong(fine_level, coarse_level, cycle_dataloader)
        fine_level.restrict(fine_level, coarse_level, cycle_dataloader)
        self.iterate_on_level(dataloader, level_ind+1)

        # postsmoothing
        cycle_dataloader = self.subsetloader.get_subset_dataloader(dataloader)
        fine_level = self.levels[level_ind]
        coarse_level = self.levels[level_ind + 1]
        fine_level.prolong(fine_level, coarse_level, cycle_dataloader)
        fine_level.postsmooth(fine_level.net, cycle_dataloader)

    def run(self, dataloader):
        """W-cycle training method

        @param dataloader
        @param verbose

        """
        self.log_cycling_header()

        # Iteratively restrict each level's grid
        for cycle in range(0, self.cycles):
            if verbose:
                printer.print_cycle_status(self, cycle)
                
            self.iterate_on_level(dataloader, 0)
            
            if self.validation_callbacks is not None:
                for vc in self.validation_callbacks:
                    vc(self.levels, cycle)


def FMG(MultilevelCycle):
    pass
    # TODO
