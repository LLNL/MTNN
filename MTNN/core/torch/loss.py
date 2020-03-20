"""
Torch Loss function dispatch table
See documentation: https://pytorch.org/docs/stable/nn.html#loss-functions

"""
import torch.nn as nn

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
