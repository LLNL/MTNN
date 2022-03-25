from MTNN.utils import logger


class Level:
    """A Level in a multilevel hierarchy contains a neural network as well
       as transfer operators between it and the next level.

    """
    def __init__(self, id: int, net, smoother, prolongation, restriction,
                 num_smoothing_passes, corrector=None):
        """
        @param id level id (assumed to be unique)
        @param net A neural network
        @param smoother smoothing method
        @param prolongation prolongation method
        @param restriction restriction method
        @param num_smoothing_passes number of passes to execute through current dataloader at each smoothing step
        @param corrector tau corrector, used for computing FAS tau correction
        """
        self.net = net
        self.id = id
        self.smoother = smoother
        self.prolongation = prolongation
        self.restriction = restriction
        self.corrector = corrector
        self.num_smoothing_passes = num_smoothing_passes
        self.l2_info = None
        self.log = logger.get_MTNN_logger()

        self.interpolation_data = None
        # Lhs
        self.Winit = None
        self.Binit = None

    def presmooth(self, model, dataloader, verbose=False):
        try:
            self.log.info("PRESMOOTHING {}".format(self.smoother.__class__.__name__).center(80, '-'))
            self.smoother.apply(model, dataloader, self.num_smoothing_passes, tau=self.corrector,
                                   l2_info = self.l2_info, verbose=verbose)
        except Exception:
            raise

    def postsmooth(self, model, dataloader, verbose=False):
        try:
            self.log.info("POSTSMOOTHING {}".format(self.smoother.__class__.__name__).center(80, '-'))
            self.smoother.apply(model, dataloader, self.num_smoothing_passes, tau=self.corrector,
                                    l2_info = self.l2_info, verbose=verbose)
        except Exception:
           raise

    def coarse_solve(self, model, dataloader, verbose=False):
        try:
            self.log.info("COARSEST SMOOTHING {}".format(self.smoother.__class__.__name__).center(80, '-'))
            self.smoother.apply(model, dataloader, self.num_smoothing_passes, tau=self.corrector,
                                l2_info = self.l2_info,verbose=verbose)
        except Exception:
            raise

    def prolong(self, fine_level, coarse_level, dataloader, verbose=False):
        try:
            self.log.info("PROLONGATION {}".format(self.prolongation.__class__.__name__).center(80, '-'))
            self.prolongation.apply(fine_level, coarse_level, dataloader, verbose)
        except Exception:
            raise

    def restrict(self, fine_level, coarse_level, dataloader, verbose=False):
        try:
            self.log.info("RESTRICTION {}".format(self.restriction.__class__.__name__).center(80, '-'))
            self.restriction.apply(fine_level, coarse_level,  dataloader,  verbose)
        except Exception:
            raise

    def view(self):
        """Logs level attributes"""
        for atr in self.__dict__:
            atrval = self.__dict__[atr]
            if type(atrval) in (int, float, str, list, bool):
                self.log.warning(f"\t{atr}: \t{atrval} ")
            else:
                self.log.warning(f"\t{atr}: \t{self.__dict__[atr].__class__.__name__}")


