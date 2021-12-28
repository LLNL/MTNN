# PyTorch
import torch.optim as optim

# local
from MTNN.utils import logger, deviceloader

# Public
__all__ = ['SGDSmoother']


class SGDSmoother:
    """A typical SGD optimizer as is used in classical neural network
    training. This is used during the "smoothing" step of the
    multilevel iteration, which is the step that does some actual
    training work on that level of the hierarchy before passing
    learned information to coarser or finer levels via restriction or
    prolongation.

    """
    def __init__(self, loss_fn, learning_rate, momentum, weight_decay):
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = None
        self.momentum_data = None
        self.log = logger.get_MTNN_logger()

    def set_momentum(self, momentum_data):
        self.momentum_data = momentum_data
        self.optimizer = None

    def initialize_smoother(self, sgd_params):
        # Create optimizer if not already there
        if self.optimizer is None:
            self.optimizer = optim.SGD(sgd_params,
                                       lr = self.learning_rate,
                                       momentum = self.momentum,
                                       weight_decay = self.weight_decay)

        # If updated momentum data available, use that
        if self.momentum_data is not None:
            # Insert momentum data
            for i in range(0, len(self.optimizer.param_groups[0]['params']), 2):
                self.optimizer.state[self.optimizer.param_groups[0]['params'][i]]['momentum_buffer'] = self.momentum_data[i]
                self.optimizer.state[self.optimizer.param_groups[0]['params'][i+1]]['momentum_buffer'] = self.momentum_data[i+1]
            self.momentum_data = None

    def log_iteration(self, loss, batch_ind, dataloader):
        self.log.info("\t{} / {} \t\tLoss: {}".format((batch_ind+1) * dataloader.batch_size,
                                                      len(dataloader) * dataloader.batch_size,
                                                      loss.item()))
        

    def apply(self, model, dataloader, num_smoothing_passes, tau=None, l2_info = None, verbose=False) -> None:
        """Apply SGD optimizer. Optionally apply tau correction if tau_corrector is given.

        @param model Neural network to train

        @param dataloader <DataLoader> PyTorch Dataloader 

        @param num_smoothing_passes Number of times through the dataloader

        @param tau <BaseTauCorrector> Tau corrector. If $w$ is the
        unrolled vector of all learnable parameters in the model and
        $\tau$ is a vector of the same length, the tau corrector adds
        a term of the form $-w^T \tau$ to the objective function. The
        tau corrector, not used at the finest level, alters the coarse
        gradient so that, immediately after restriction, it is a
        restricted analogue of the fine gradient. This enables the
        coarse model to "learn like" the fine level, at least for the
        first few iterations.

        @param verbose <bool> Prints statistics/output to standard out

        Returns:
            None

        """
        self.initialize_smoother(model.parameters())
        
        for pass_ind in range(num_smoothing_passes):
            for batch_idx, mini_batch_data in enumerate(dataloader):
                input_data, target_data = deviceloader.load_data(mini_batch_data, model.device)
                self.loss_fn.to(model.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                                
                # Forward
                outputs = model(input_data)
                loss = self.loss_fn(outputs, target_data)
                    
                # Apply Tau Correction if present
                if tau:
                    tau.correct(model, loss, batch_idx, len(dataloader), verbose)

                # Backward
                loss.backward()
                
                self.optimizer.step()
                self.log_iteration(loss, batch_idx, dataloader)
                # if verbose:
                #     printer.print_smoother(loss, batch_idx, dataloader, tau)








