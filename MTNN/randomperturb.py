"""
Prolongation Operators
"""
# Public API
__all__ = ["RandomPerturbation"]


class RandomPerturbation:
    def __init__(self, prolongation=None):
        self.prolongation = prolongation

    def apply(self, sourcemodel):
        if self.prolongation == "randomperturb":  # TODO: better switch statement
            for name, param in enumerate(sourcemodel.state_dict()):
                if "weight" in param:
                    pass
                    # Transform the parameter

                    # Update the parameter.
                   # sourcemodel.state_dict[name].copy_(transformed_param)

        # TODO: lower triangular
        # TODO: randomsplit

