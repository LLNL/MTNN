"""
Trainer
"""
# local
import MTNN.utils as utils

default_save_path = utils.make_default_path(dir= "models", ext= ".pt")

class MultigridTrainer:
    """
    Takes a model and applies some optimizer.
    """
    def __init__(self, dataloader, verbose=False, save=False,
                 save_path=default_save_path, load=False, load_path=""):
        self.dataloader = dataloader
        self.verbose = verbose
        self.save = save
        self.save_path = save_path
        self.load = load
        self.load_path = load_path

    def train(self, model, optimizer, cycles):
        """

        Args:
            model: <MTNN.BaseModel>
            optimizer: <MTNN.core.optimizer.multigrid.BaseMultigridHierarchy>
            cycles: <int> Number of iterations per level

        Returns:

        """


        for i in range(cycles):
            if self.verbose:
                print(f"Cycle {i + 1}/{cycles}")

            trained_model = optimizer.run(model, self)

        return trained_model


