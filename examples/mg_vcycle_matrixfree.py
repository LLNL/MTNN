"""
Example of FAS VCycle
"""
import time
from collections import namedtuple

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

import sys

sys.path.append("../../mtnnpython")

# local
from MTNN.core.components import data, models, subsetloader
from MTNN.core.multigrid.operators import smoother, restriction, prolongation, tau_corrector
from MTNN.core.multigrid.interpolation import similarity, matching, coarsener
from MTNN.core.alg import trainer, evaluator
import MTNN.core.multigrid.scheme as mg

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_printoptions(precision = 5)

# =======================================
# Set-up
# =====================================
# test_data = data.TestData(trainbatch_size=10, testbatch_size=10)
fake_data = data.FakeData(imagesize = (1, 2, 2),
                          num_classes = 2,
                          trainset_size = 10,
                          trainbatch_size = 1,
                          testset_size = 10,
                          testbatch_size = 10)
# vcycle_data = data.CycleLoader(fake_data.trainloader, alt=True)
net = models.MultiLinearNet([4, 3, 2], F.relu, F.log_softmax)

# With Mnist
# data = data.MnistData(trainbatch_size=100, testbatch_size=100)
# net = models.MultiLinearNet([784, 50, 25, 10], F.relu, F.log_softmax)


# =====================================
# Multigrid Hierarchy
# =====================================
# Specify optimizer and data loader type
SGDparams = namedtuple("SGDparams", ["lr", "momentum", "l2_decay"])
subsetloader_type = subsetloader.NextKLoader


# Specify the multigrid operators
prolongation_op = prolongation.MatrixFreePairwiseAggProlongation
restriction_op = restriction.MatrixFreePairwiseAggRestriction

similarity = similarity.StandardSimilarity
matching = matching.HeavyEdgeMatching
coarsener = coarsener.PairwiseAggCoarsener
tau = tau_corrector.BasicTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = 3
FAS_levels = []

pre_smooth_epochs = [3, 3, 3]
post_smooth_epochs = [3, 3, 3]
coarse_smoother_epochs = [3, 3, 3]

# Control number of epochs and learning rate per level
for level_idx in range(0, num_levels):
    if level_idx == 0:
        optim_params = SGDparams(lr = 0.01, momentum = 0.01, l2_decay = 1e-2)
        loss_fn = nn.CrossEntropyLoss()
    elif level_idx == 1:
        optim_params = SGDparams(lr = 0.01, momentum = 0.01, l2_decay = 1e-2)
        loss_fn = nn.NLLLoss()
    else:
        optim_params = SGDparams(lr = 0.01, momentum = 0.01, l2_decay = 1e-2)
        loss_fn = nn.NLLLoss()

    # Asymmetric smoothing
    pre_smooth_epochs = [1, 2, 3]
    post_smooth_epochs = [1, 2, 3]

    pre_sgd_smoother = smoother.SGDSmoother(loss_fn = loss_fn,
                                            optim_params = optim_params,
                                            num_epochs = pre_smooth_epochs[level_idx],
                                            log_interval = 10)  # Log interval is 0 by default

    post_sgd_smoother = smoother.SGDSmoother(loss_fn = loss_fn,
                                             optim_params = optim_params,
                                             num_epochs = post_smooth_epochs[level_idx],
                                             log_interval = 10)

    coarse_sgd_smoother = smoother.SGDSmoother(loss_fn = loss_fn,
                                               optim_params = optim_params,
                                               num_epochs = coarse_smoother_epochs[level_idx],
                                               log_interval = 10)

    aLevel = mg.Level(id = level_idx,
                      presmoother = pre_sgd_smoother,
                      postsmoother = post_sgd_smoother,
                      prolongation = prolongation_op(),
                      restriction = restriction_op(coarsener(matching(similarity()))),
                      coarsegrid_solver = coarse_sgd_smoother)
                      #corrector = tau(loss_fn, log_interval = 10))

    FAS_levels.append(aLevel)

mg_scheme = mg.VCycle(FAS_levels,
                      cycles = 3,
                      subsetloader = subsetloader_type(num_minibatches = 10),
                      coarsen_cyclesteps=1,
                      learning_rate_cyclestep = 2,
                      learning_rate_factor=0.5)

training_alg = trainer.MultigridTrainer(scheme = mg_scheme,
                                        verbose = 'debug', # brief or debug
                                        log = True,
                                        save = False,
                                        load = False)


# =====================================
# Train
# =====================================
print('Starting Training')
start = time.perf_counter()
trained_model = training_alg.train(model = net, dataloader = fake_data.trainloader)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))
# =====================================
# Test
# =====================================
evaluator = evaluator.CategoricalEvaluator()
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model = net, dataloader = fake_data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop - start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
