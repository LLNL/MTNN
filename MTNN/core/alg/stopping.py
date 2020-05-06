"""
Holds Multigrid stopping measures
"""

# Functions or Class?




class Stopper:

    def __init__(self):
        self.TotalLoss = None
        self.RelativeLoss = None
        self.NumCycles = None
        self.NumEpochs = None


def ByTotalLoss():
    pass
    # TODO: Fill in


def ByRelativeLoss():
    pass
    # TODO: Fill in


def ByNumIterations():
    pass
    # TODO: Fill in


def ByNumEpochs(num=0):
    stopper = Stopper()
    stopper.NumEpochs = num
    return stopper


