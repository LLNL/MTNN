from MTNN.utils import logger


log = logger.get_logger(__name__, write_to_file =True)

__all__ = ['printSmoother',
           'printLevelStats',
           'printModel']


def printSmoother(epoch:int, loss:int, batch_idx:int, dataloader, stopper, log_interval:int):
    if ((batch_idx) * dataloader.batch_size % log_interval) == 0 and batch_idx != 0:
        log.info(f"Epoch: {epoch}/{stopper.max_epochs}"
                 f"\t{batch_idx * dataloader.batch_size} / {len(dataloader.dataset)}"
                 f"\t\tLoss: {loss.item()}")
    elif batch_idx + 1 == len(dataloader):
        log.info(f"Epoch: {epoch}/{stopper.max_epochs}"
                 f"\t{(batch_idx + 1) * dataloader.batch_size} / {len(dataloader.dataset)}"
                 f"\t\tLoss: {loss.item()}")


def printLevelStats( level_idx: int, num_levels: int, msg="",):
    """
    Args:
        msg:
        level_idx:
        num_levels:

    Returns:

    """
    log.info(f"{msg} Level {level_idx}: {level_idx + 1}/{num_levels}")


def printModel(model, msg="", **options):
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
    except AttributeError:
        print(f"Model is unintialized.")
