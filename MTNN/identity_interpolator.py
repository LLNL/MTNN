class IdentityInterpolator():
    """Identity Interpolation Operator

    Copy model weights.
    """
    
    def __init__(self):
        pass

    def apply(self, source_model):
        return source_model
