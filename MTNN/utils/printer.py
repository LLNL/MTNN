# standard
import collections as col
import numpy as np
from numpy import linalg as LA

from MTNN.utils import logger
log = logger.get_logger(__name__, write_to_file =True)

__all__ = ['printSmoother',
           'printLevelStats',
           'printModel']


def printSmoother(epoch:int, loss:int, batch_idx:int, dataloader, stopper, log_interval:int) -> None:
    """ Print based on specified logging interval.
    Args:
        log_interval: <int> Specifies batch intervals to log/print-out
    """
    if ((batch_idx) * dataloader.batch_size % log_interval) == 0 and batch_idx != 0:
        log.info(f"Epoch: {epoch}/{stopper.max_epochs}"
                 f"\t{batch_idx * dataloader.batch_size} / {len(dataloader.dataset)}"
                 f"\t\tLoss: {loss.item()}")
    elif batch_idx + 1 == len(dataloader):
        log.info(f"Epoch: {epoch}/{stopper.max_epochs}"
                 f"\t{(batch_idx + 1) * dataloader.batch_size} / {len(dataloader.dataset)}"
                 f"\t\tLoss: {loss.item()}")



def printLevelStats( level_idx: int, num_levels: int, msg="",) -> None:
    """
    Args:
        level_idx: <int> Level Id
        num_levels: <int> Number of Levels
        msg: <str> Message to print
    """
    log.info(f"{msg} Level {level_idx}: {level_idx + 1}/{num_levels}")


def printLevelInfo(levels: list) -> None:
    for idx, level in enumerate(levels):
        log.info(f"Level {idx}")
        level.view()


def printModel(model, msg="", **options) -> None:
    # TODO: Refactor
    try:
        if 'val' in options and options['val']:
            log.info(f"{msg} \n Model Parameters")
            for layer_idx, layer in enumerate(model.layers):
                log.info(f"\tLAYER {layer_idx}")
                log.info(f"\t\tWEIGHTS  \t{layer.weight.data}")
                log.info(f"\t\tBIAS  \t{layer.bias.data}")

        if 'dim' in options and options['dim']:
            log.info(f"{msg} \n Model Dimensions")
            for layer_idx, layer in enumerate(model.layers):
                log.info(f"\tLAYER {layer_idx} WEIGHT DIM\t{layer.weight.size()} \tBIAS DIM {layer.bias.size()}")

        if 'grad' in options and options['grad']:
            log.info(f"{msg} \n Model Gradients")
            for layer_idx, layer in enumerate(model.layers):
                log.info(f"\tLAYER {layer_idx}")
                log.info(f"\t\tWEIGHTS  \t{layer.weight.grad}")
                log.info(f"\t\tBIAS  \t{layer.bias.grad}")


    except AttributeError:
        log.warning(f"Net is empty.")

def printGradNorm(loss, weights, bias) -> None:
    # TODO
    # Yield  gradient norm
    # total_loss -= Tr(rhsW' * W) + Tr(rhsB' * B)
    # grad -= rh
    if weights and bias:
        RHS = col.namedtuple('rhs', ['weights', 'bias'])
        rhs = RHS(weights, bias)

        total_loss = loss.item()
        norm_dW = 0
        norm_dB = 0
        num_layers = len(self.layers)
        for layer_id in range(num_layers):
            with torch.no_grad():
                dW = np.copy(self.layers[layer_id].weight.grad.detach().numpy())
                dB = np.copy(self.layers[layer_id].bias.grad.detach().numpy().reshape(-1, 1))
                if rhs.weights:
                    total_loss -= np.sum(rhs.weights[layer_id] * self.layers[layer_id].weight.detach().numpy())
                    dW -= rhs.weights[layer_id]
                if rhs.bias:
                    total_loss -= np.sum(rhs.bias[layer_id] * self.layers[layer_id].bias.detach().numpy())
                    dB -= rhs.bias[layer_id]
                norm_dW += LA.norm(dW, 'fro') ** 2
                norm_dB += LA.norm(dB, 'fro') ** 2
        norm_dW = norm_dW ** (0.5)
        norm_dB = norm_dB ** (0.5)

        yield total_loss, norm_dW, norm_dB
