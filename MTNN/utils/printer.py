"""Holds printing functions"""
#local
from MTNN.utils import logger
log = logger.get_logger(__name__, write_to_file =True)

# Public
__all__ = ['print_smoother',
           'print_cycleheader',
           'print_levelstats',
           'print_level',
           'print_model',
           'print_tau']


def format_header(text: str, width=100, border="=") -> str:
    """
    Centers string with a border.

    Args:
        text: <str> String to format
        width: <int> Length of string
        border: <str> Character to fill missing space on either side.
    Returns:
        Formatted string
    """
    return f"{text}".center(width, border)


def print_smoother(loss:int, batch_idx:int, dataloader, log_interval: int, tau:None) -> None:
    """
    Print minibatch and loss based on specified logging interval.

    Args:
        loss: <Tensor>
        batch_idx: <int>
        dataloader: <MTNN.core.components.data> subclass of BaseDataLoader
        log_interval: <int> Specifies batch intervals to log/print-out
    """
    batch_idx = batch_idx + 1
    if ((batch_idx * dataloader.batch_size) % log_interval) == 0:
        log.info(f"\t{(batch_idx) * dataloader.batch_size} / {len(dataloader.dataset)}"
                 f"\t\tLoss: {loss.item()}")


def print_cycleheader(cycleclass) -> None:
    """
    Print Cycle class name.

    Args:
        cycleclass: <MTNN.core.multigrid.scheme> subclass of BaseMultigridScheme

    Returns:
    """
    log.info(format_header(f"Applying {cycleclass.__class__.__name__}"))
    log.info(f"Number of Cycles: {cycleclass.cycles}")


def print_cycle_status(cycleclass, cycle=None) -> None:
    """
     Print Cycle class name.

     Args:
         cycleclass: <MTNN.core.multigrid.scheme> subclass of BaseMultigridScheme

     Returns:
     """
    log.info(format_header(f"CYCLE {cycle + 1} /{cycleclass.cycles}"))


def print_levelstats(cycle, maxcycles, level_idx: int, num_levels: int, msg="", ) -> None:
    """
    Print Level ID and number of remain levels to iterate through.
    Args:
        level_idx: <int> Level ID
        num_levels: <int> Number of Levels
        msg: <str> Message to print
    Returns:
        NOne
    """
    log.info(f"{msg} Cycle {cycle +1}/{maxcycles} Level {level_idx}: {level_idx + 1}/{num_levels}")


def print_level(levels: list) -> None:
    """
    Given a list of level, logs each Level instance's id and  attributes

    Args:
        levels: List of <MTNN.core.multigrid.scheme> Level objects

    Returns:
        None
    """
    for idx, level in enumerate(levels):
        log.info(f"Level {idx}")
        level.view()


def print_model(model, msg="", **options) -> None:
    # TODO: Refactor
    """
    For debugging. Prints model parameters.

    Args:
        model: <MTNN.core.components.model>
        msg: <str>
        **options: <bool>

    Returns:
        None
    """
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


def print_tau(tau, loss, msg="") -> None:
    """
    Prints tau_corrector class with the loss.

    Args:
        tau: <MTNN.core.multigrid.operators.tau_corrector>
        loss: <Tensor>
        msg: <str>

    Returns:
        None
    """
    log.info(f"{msg}{tau.__class__.__name__} Loss = {loss}")


def printGradNorm(loss, weights, bias) -> None:
    # TODO
    """
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
        """
