"""MTNN/torch_consts.py
Stores PyTorch Functions used by Model
See: https://pytorch.org/docs/stable/optim.html#
Note: Keys are defined by their respective PyTorch Function Name
# TODO: Make keys case-insensitive
"""
# pytorch
import torch.nn as nn
import torch.optim as optim


######################################
# Optimization Function Dispatch Table
######################################
# See documentation for optimizers here:https://pytorch.org/docs/stable/optim.html
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
    opt = optim.SGD(model_parameters, lr=learning_rate, momentum=momentum)
    return opt
######################################
# Loss Function Dispatch Table
######################################
LOSS = {
    # See documentation for these functions here: https://pytorch.org/docs/stable/nn.html#loss-functions
    "l1loss": nn.L1Loss(),
    "bceloss": nn.BCELoss(),
    "bcewithlogitsloss": nn.BCEWithLogitsLoss(),
    "cosineembeddingloss": nn.CosineEmbeddingLoss(),
    "crossentropyloss": nn.CrossEntropyLoss(),
    "ctcloss": nn.CTCLoss(),
    "hingeembeddingloss": nn.HingeEmbeddingLoss(),
    "kldivLoss": nn.KLDivLoss(),
    "marginrankingloss": nn.MarginRankingLoss(),
    "mseloss": nn.MSELoss(),
    "multilabelmarginloss": nn.MultiLabelMarginLoss,
    "multilabelsoftmarginloss": nn.MultiLabelSoftMarginLoss(),
    "multimarginloss": nn.MultiMarginLoss(),
    "nllloss": nn.NLLLoss(),
    "poissonnllloss": nn.PoissonNLLLoss(),
    "smoothl1loss": nn.SmoothL1Loss(),
    "softmarginloss": nn.SoftMarginLoss(),
    "tripletmarginloss": nn.TripletMarginLoss(),
}

######################################
# Activation Function Dispatch Table
######################################
ACTIVATIONS = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "softmax": nn.Softmax()
}
