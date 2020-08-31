"""
Trainer
"""
# local
import MTNN.utils.file as file
import MTNN.utils.logger as logger

log = logger.get_logger(__name__, write_to_file=True)

default_save_path = file.make_default_path(dir= "models", ext= ".pt")


class Session:
    "Store model and trainer settings"
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer


class MultigridTrainer:
    """
    Takes a model and applies some multigrid scheme <core.multigrid.multigrid>
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
        self.callbacks = None #TODO:  design pattern from Keras Callback API https://keras.io/api/callbacks/

    def train(self, model, multigrid, cycles: int):
        """

        Args:
            model: <MTNN.BaseModel>
            multigrid: <MTNN.core.multigrid.multigrid.BaseMultigridHierarchy>
            cycles: <int> Number of iteratons through the multigrid hiearchy

        Returns:

        """

        session = Session(model, self)
        multigrid.setup(session.model)
        trained_model = multigrid.run(session, cycles)

        return trained_model


