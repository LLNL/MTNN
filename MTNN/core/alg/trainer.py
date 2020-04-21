"""
Trainer
"""

# local
import MTNN.utils as utils

SAVE_PATH = "./models.path"


class MultigridTrainer:
    """
    Takes a model and applies some optimizer.
    """
    def __init__(self, dataloader, train_batch_size: int, verbose=False):
        self.dataloader = dataloader
        self.verbose = verbose

    def train(self, model, optimizer):

        for i, data in enumerate(self.dataloader, 0):

            # Show status bar
            if self.verbose:
                total_work = len(self.dataloader)
                utils.progressbar(i, total_work, status="Training")

            model = optimizer.run(model, data)

        return model


