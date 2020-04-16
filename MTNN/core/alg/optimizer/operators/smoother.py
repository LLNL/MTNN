"""
Holds Smoothers
"""
class BaseAlgSmoother():
    """Training Algorithm Smoother
    
    Applies a given training algorithm to the model.

    Attributes:
    alg_: Any object with a suitable "train" method
    stopping_: A stopping criterion suitable for the algorithm
    """

    def __init__(self, alg, stopping):
        self.alg_ = alg
        self.stopping_ = stopping

    def smooth(self, model, data, obj_func):
        self.alg_.train(model, data, obj_func, self.stopping_)


class SGDSmoother(BaseAlgSmoother):

    def __init__(self):
        pass