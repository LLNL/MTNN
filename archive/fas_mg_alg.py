import pdb
import MTNN

class FASMG():
   """FAS Multigrid Algorithm

   Attributes:
   levels: A list of levels
   """
   def __init__(self, levels):
      self.levels = levels

   def Vtrain(self, net, dataset, resetup=True):
      num_levels = len(self.levels)
      self.levels[0].net = net
      # Down cycle
      for lev_id in range(num_levels-1):
         fine_level = self.levels[lev_id]
         coarse_level = self.levels[lev_id+1]
         fine_level.smooth(1, dataset)
         if resetup:
            fine_level.coarsener.coarsen(fine_level.net)
            fine_level.restriction.setup(fine_level, coarse_level)
         fine_level.restriction.restrict(fine_level, coarse_level, dataset)

      # Do the last level's solving/smoothing:
      self.levels[-1].smooth(3, dataset)

      # Up cycle
      for lev_id in range(num_levels-2,-1,-1):
         fine_level = self.levels[lev_id]
         coarse_level = self.levels[lev_id+1]
         fine_level.prolongation.prolongate(coarse_level, fine_level)
         fine_level.smooth(2, dataset)

      return self.levels[0].net

