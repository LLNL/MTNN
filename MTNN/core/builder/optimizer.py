"""
Optimizer Class Factory
"""

import MTNN.core.torch as torch

def build(confpath: str, model_parameters):
    """
    Uses an optimization dispatch table to instantiate the optimization specified by the provided configuration file.
    Args:
        confpath (str):  absolute file path to the YAML configuration file
        model_parameters:

    Returns:
        optimizer <torch.optim>
    """
    conf = config_reader.YamlConfig(confpath)
    optimization = conf.optimization
    learning_rate = conf.learning_rate
    momentum = conf.momentum

    optimization_dispatch_table = {
        "Adadelta": torch_consts.Adadelta(),
        "Adagrad": torch_consts.Adagrad(),
        "Adam": torch_consts.Adam(),
        "AdamW": torch_consts.AdamW(),
        "SparseAdam": torch_consts.SparseAdam(),
        "Adamax": torch_consts.Adamax(),
        "ASGD": torch_consts.ASGD(),
        "LBFGS": torch_consts.LBFGS(),
        "RMSprop": torch_consts.RMSprop(),
        "Rprop": torch_consts.Rprop(),
        "SGD": torch_consts.SGD(model_parameters, learning_rate, momentum)
    }

    optimizer = optimization_dispatch_table[optimization]
    return optimizer




