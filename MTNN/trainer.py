""" MTNN/trainer.py
Reads from YAML configuration file and returns a PyTorch optimization object
"""
# pytorch
import torch.optim as optim

# local source
import MTNN.config_reader as reader

######################################
# Optimization functions
######################################
# TODO: Fill in as needed
def Adadelta():
    pass


def Adagrad():
    pass


def Adam():
    pass


def AdamW():
    pass


def SparseAdam():
    pass


def Adamax():
    pass


def ASGD():
    pass


def LBFGS():
    pass


def RMSprop():
    pass


def Rprop():
    pass


def SGD(model_parameters, learning_rate, momentum):
    opt = optim.SGD(model_parameters, lr = learning_rate, momentum = momentum)
    return opt


def build_optimizer(confpath: str, model_parameters):
    """
    Uses an optimization dispatch table to instantiate the optimization specified by the provided configuration file.
    Args:
        confpath (str):  absolute file path to the YAML configuration file
        model_parameters:

    Returns:
        optimizer <torch.optim>
    """
    conf = reader.YamlConfig(confpath)
    optimization = conf.optimization
    learning_rate = conf.learning_rate
    momentum = conf.momentum

    optimization_dispatch_table = {
        "Adadelta": Adadelta(),
        "Adagrad": Adagrad(),
        "Adam": Adam(),
        "AdamW": AdamW(),
        "SparseAdam": SparseAdam(),
        "Adamax": Adamax(),
        "ASGD": ASGD(),
        "LBFGS": LBFGS(),
        "RMSprop": RMSprop(),
        "Rprop": Rprop(),
        "SGD": SGD(model_parameters, learning_rate, momentum)
    }

    optimizer = optimization_dispatch_table[optimization]
    return optimizer




