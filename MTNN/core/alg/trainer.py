"""
Trainer
"""
# local
import MTNN.core.components.models as mtnnmodel
import MTNN.utils.file as file
import MTNN.utils.logger as logger

log = logger.get_logger(__name__, write_to_file=True)

default_save_path = file.make_default_path(dir= "models", ext= ".pt")


class MultigridTrainer:
    """
    Takes a model and applies some multigrid scheme <core.multigrid.multigrid>
    """

    def __init__(self, scheme, verbose=False, log=False, save=False,
                 save_path=default_save_path, load=False, load_path=""):
        self.scheme = scheme
        self.verbose = verbose  # print to stdout
        self.log = log  # saves output to file
        self.save = save  # checkpoints model
        self.save_path = save_path
        self.load = load
        self.load_path = load_path

    def train(self, model, dataloader):
        """

        Args:
            model: <MTNN.core.components.models> subclass of BaseModel
            dataloader: <MTNN.core.components.data> subclass of BaseDataLoader
        Returns:

        """
        assert isinstance(model, mtnnmodel._BaseModel)
        self.scheme.setup(model)
        trained_model = self.scheme.run(model, dataloader, self)

        return trained_model


