"""
Trainer
"""
# standard
from pathlib import Path, PurePath

# local
import MTNN.utils as utils

# TODO: Add default save path with filename + date
DEFAULT_SAVE_DIR = PurePath.joinpath(Path.cwd(), Path("./model/"))

class MultigridTrainer:
    """
    Takes a model and applies some optimizer.
    """
    def __init__(self, dataloader, verbose=False, save=False,
                 save_path="", load=False, load_path=""):
        self.dataloader = dataloader
        self.verbose = verbose
        self.save = save
        self.save_path = utils.make_path(DEFAULT_SAVE_DIR, save_path)
        self.load = load
        self.load_path = load_path

    def train(self, model, optimizer, cycles):
        for i in range(cycles):
            if self.verbose:
                print(f"Cycle {i + 1}/{cycles}")
            trained_model = optimizer.run(model, self)

        return trained_model


