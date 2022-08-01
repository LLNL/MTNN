"""
Holds Tau correction definitions
"""
from abc import ABC, abstractmethod
import copy
import torch
from MTNN.utils import logger

# Public
__all__ = ['NullTau',
           'WholeSetTau',
           'MinibatchTau']


def put_tau_together(fine_tau, fine_grad, coarse_grad, ops):
    """ Numerical work to construct the tau correction vector.

    @param fine_tau <ParamVector>  The tau correction vector from the next-finer level. [f^h]
    @param fine_grad <ParamVector> The gradient from the next-finer level. [A^h(u)]
    @param coarse_grad <ParamVector> The gradient from the current, coarse level. [A^{2h}(R*u)]
    @param ops <TransferOps> The transfer operators from the fine to the current, coarse level.

    Output:
    <ParamVector> The tau correction vector for the current, coarse level.
    """
    # Construct [f^h - A^h(u)]
    diff_params = fine_tau - fine_grad

    # Apply restriction to construct R * [f^h - A^h(u)]
    coarse_diff_params = ops @ diff_params

    # Add final term to construct R * [f^h - A^h(u^h)] + A^{2h}(R*u^h)
    return coarse_diff_params + coarse_grad
        

###################################################################
# Interface
####################################################################

class BaseTauCorrector(ABC):
    """In the Full Approximation Scheme, originally for nonlinear systems
    of equations, the tau correction is a vector added to the
    right-hand side of coarse-level equations that enables the coarse
    solver to behave more like a coarsened version of the fine
    solver. In the case of FAS for optimization as being done here,
    the tau correction has the effect of altering the gradient at the
    coarse level.

    We choose our tau correction vector such that it replaces the
    initial (ie immediately after restriction) coarse gradient with
    one that is a restricted analogue of the fine gradient. This means
    that, initially, the coarse optimizer will behave like it is
    operating on a restricted subspace on the fine level.

    The tau correction vector is chosen at restriction and is constant
    until the next V-cycle, so it doesn't alter the higher-order
    derivatives of the optimizer. Thus, this interpretation of the tau
    correction weakens at smoothing iterations progress. Multilevel
    training tends to be successful when BOTH the fine level performs
    well and the tau-shifted coarse levels perform well.

    One interpretation of this is as a kind of regularization: We are
    looking for a point in the parameter space in which the gradient
    at the fine level is near zero, the Hessian at the fine level is
    postive-definite, AND the Hessian at all coarser levels of the
    hierarchy are also positive-definite.

    """

    
    def __init__(self, loss_fn, gradient_extractor):
        """
        @param loss_fn The loss function to use on training data in
        computing gradients.  

        @gradient_extractor <GradientExtractor>
        """
        self.loss_fn = loss_fn
        self.gradient_extractor = gradient_extractor

    @abstractmethod
    def get_fine_tau(self, batch_idx = None, mini_dataloader = None):
        """ Get the tau computed at this level for a given minibatch."""
        raise NotImplementedError

    @abstractmethod
    def compute_tau(self, coarse_level, fine_level, dataloader, operators, **kwargs):
        """Computes residual tau of the coarse-level"""
        raise NotImplementedError

    @abstractmethod
    def correct(self, model, loss, num_batches, **kwargs):
        """Returns corrected loss """
        raise NotImplementedError

###################################################################
# Concrete subclasses
####################################################################

class NullTau(BaseTauCorrector):
    """The NullTau is always 0.

    This is the same as having no tau corrector. This, in some sense,
    imposes a stricter regularization on the training process: in
    addition to searching for a point in the parameter space in which
    the fine gradient is near 0 and the Hessians at all levels are
    positive-definite, we ALSO are looking for a point in the
    parameter space in which all coarse gradients are
    positive-definite too.

    """
    def __init__(self, loss_fn, gradient_extractor):
        super().__init__(loss_fn, gradient_extractor)

    def get_fine_tau(self, batch_idx = None, mini_dataloader = None):
        return 0.0

    def compute_tau(self, coarse_level, fine_level, dataloader, operators):
        pass

    def correct(self, model, loss, batch_idx, num_batches, verbose = False):
        pass

class WholeSetTau(BaseTauCorrector):
    """WholeSetTau computes the gradient over all examples in a given
    dataloader, resulting in the most accurate tau vector. If passed
    the entire training set, the tau corrector is deterministic.

    This CAN compute a tau correction over the whole training set,
    though that is likely impractical for large training
    sets. However, our V-cycle uses a SubsetDataloader to extract a
    new training subset, usually consisting of a few (2-20)
    minibatches, at each level of the V-cycle, and this is typically
    what is passed to the WholeSetTau. The result is a tau that is
    much faster to compute, though is only an approximation to the
    true whole-dataset tau.

    """
    def __init__(self, loss_fn, gradient_extractor, scaling = 1.0):
        super().__init__(loss_fn, gradient_extractor)
        self.tau = None
        self.scaling = scaling

    def get_fine_tau(self, batch_idx = None, mini_dataloader = None):
        if self.tau is None:
            return 0.0
        else:
            return self.tau

    def compute_tau(self, coarse_level, fine_level, dataloader, operators, verbose=False):
        fine_tau = fine_level.corrector.get_fine_tau() # of type ParamVector
        fine_grad = self.gradient_extractor.extract_from_network(fine_level, dataloader, self.loss_fn) #ParamVector
        coarse_grad = self.gradient_extractor.extract_from_network(coarse_level, dataloader, self.loss_fn)

        self.tau = put_tau_together(fine_tau, fine_grad, coarse_grad, operators)
        self.tau_network_format = copy.deepcopy(self.tau)

        # TODO: This breaks the encapsulation of the gradient
        # extractor to directly use the converter underneath. Refactor
        # to maintain encapulsation.
        self.gradient_extractor.converter.convert_MTNN_format_to_network(self.tau_network_format)

        
    def correct(self, model, loss, batch_idx, num_batches, verbose=False):
        # Normalize over number of minibatches in the dataloader
        if self.tau is not None:
            for layer_id in range(len(model.layers)):
                loss -= (self.scaling / num_batches) * \
                    torch.sum(torch.mul(model.layers[layer_id].weight, self.tau_network_format.weights[layer_id]))
                loss -= (self.scaling / num_batches) * \
                    torch.sum(torch.mul(model.layers[layer_id].bias,
                                        self.tau_network_format.biases[layer_id].reshape(-1)))


class MinibatchTau(BaseTauCorrector):
    """A tau corrector that computes a tau correction for each minibatch,
    and cycles through the corrections one at a time.
    """

    def __init__(self, loss_fn, gradient_extractor, scaling = 1.0):
        super().__init__(loss_fn, gradient_extractor)
        self.tau_array = None
        self.finer_level_corrector = None
        self.scaling = scaling

    def get_fine_tau(self, batch_idx, mini_dataloader):
        """In general we won't have already computed the tau for this minibatch,
        so recursively go down to the finer levels to build up tau."""
        if self.finer_level_corrector is None:
            return 0.0
        else:
            return self.finer_level_corrector.compute_tau_for_one_minibatch(mini_dataloader)

    def compute_tau_for_one_minibatch(self, coarse_level, fine_level, batch_idx, mini_dataloader):
        fine_tau = fine_level.corrector.get_fine_tau(batch_idx, mini_dataloader)
        fine_grad = self.gradient_extractor.extract_from_network(fine_level, mini_dataloader, self.loss_fn)
        coarse_grad = self.gradient_extractor.extract_from_network(coarse_level, mini_dataloader, self.loss_fn)
        tau = put_tau_together(fine_tau, fine_grad, coarse_grad, self.operators)
        # TODO: Fix encapsulation breaking
        self.gradient_extractor.converter.convert_MTNN_format_to_network(tau)
        return tau

    def compute_tau(self, coarse_level, fine_level, dataloader, operators, verbose=False):
        # Ensure the coarse corrector can reach back to this one,
        # needed for recursion when the hierarchy gets deeper than 2
        # levels
        self.finer_level_corrector = self
        # Storing operators also needed for recursion
        self.operators = operators

        # Clear self.tau_array for next sequence of minibatch taus by making new dict
        self.tau_array = {}
        for batch_idx, mini_batch_data in enumerate(dataloader):
            mini_dataloader = (mini_batch_data,)
            self.tau_array[batch_idx] = self.compute_tau_for_one_minibatch(
                coarse_level, fine_level, batch_idx, mini_dataloader)

    def correct(self, model, loss, batch_idx, num_batches, verbose=False):
        # TODO: Correctness here relies on the minibatch ordering not being shuffled. Fix.
        if self.tau_array is not None:
            for layer_id in range(len(model.layers)):
                loss -= self.scaling * torch.sum(torch.mul(model.layers[layer_id].weight, self.tau_array[batch_idx].weights[layer_id]))
                loss -= self.scaling * torch.sum(torch.mul(model.layers[layer_id].bias, self.tau_array[batch_idx].biases[layer_id]))
            






