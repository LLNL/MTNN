"""
Trainer
"""
# local
import MTNN.utils.file as file
import MTNN.utils.logger as logger

log = logger.get_logger(__name__, write_to_file=True)

default_save_path = file.make_default_path(dir= "models", ext= ".pt")


class MultigridTrainer:
    """
    Takes a model and applies some optimizer.
    """
    def __init__(self, dataloader, verbose=False, log=False, save=False,
                 save_path=default_save_path, load=False, load_path=""):
        self.dataloader = dataloader
        self.verbose = verbose  # print to stdout
        self.log = log  # saves output to file
        self.save = save  # checkpoints model
        self.save_path = save_path
        self.load = load
        self.load_path = load_path

    def train(self, model, multigrid, cycles):
        """

        Args:
            model: <MTNN.BaseModel>
            multigrid: <MTNN.core.optimizer.multigrid.BaseMultigridHierarchy>
            cycles: <int> Number of iterations per level

        Returns:

        """

        for i in range(cycles):
            if self.verbose:
                log.info(f"Cycle {i + 1}/{cycles}")

            trained_model = multigrid.run(model, self)

        return trained_model


