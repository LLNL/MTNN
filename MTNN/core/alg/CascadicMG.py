class Level:
    """A level in an MG hierarchy"""
    def __init__(self, presmoother=None, postsmoother=None, prolongation=None, refinement=None, coarse_solver=None):
        self.presmoother = presmoother
        self.postsmoother = postsmoother
        self.prolongation = prolongation
        self.refinement = refinement
        self.coarse_solver = coarse_solver


class CascadicMG():
    """Cascadic Multigrid Algorithm

    Attributes:
    levels: A list of levels
    """

    def __init__(self, num_levels, smoother=None, prolongation=None, refinement=None):
        self.levels = [Level(presmoother=smoother, prolongation=prolongation, refinement=refinement)] * num_levels

    def train(self, net, dataset, obj_func, criteria, expansion):
        num_levels = len(self.levels)
        for lev_id in range(num_levels-1):
            this_level = self.levels[lev_id]
        
            this_level.presmoother.smooth(net, dataset, obj_func)
            this_level.prolongation.apply(net, expansion)

            net = this_level.refinement.apply(net)
        # Do the last level's smoothing:
        self.levels[-1].presmoother.smooth(net, dataset, obj_func)
        return net
    
