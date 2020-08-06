"""
Debugging script. To be used in conjunction with fas_example_debug.py
"""
import time
from collections import namedtuple

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# local
from MTNN.core.components import data, models
from MTNN.core.multigrid.operators import smoother, prolongation, restriction, interpolator
from MTNN.core.alg import trainer, evaluator, stopping
import MTNN.core.multigrid.scheme as mg
import MTNN.utils.builder as builder


# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_printoptions(precision=5)


#=======================================
# Set-up
#=====================================
dataloader = data.FakeData(imagesize=(1, 2, 2), num_classes=2, trainbatch_size=10, testbatch_size=10)
net = models.MultiLinearNet([4, 3, 2], F.relu, F.log_softmax, weight_fill = 1, bias_fill=1)

#test_data = data.TestData(trainbatch_size=10, testbatch_size=10)
#=====================================
# Multigrid Hierarchy Components
#=====================================

# TODO: Epoch Power Control number of epochs per stopper per level
# TODO: Control the learning rate per level
# TODO: Control the l2 decay per stopper per level
SGDparams = namedtuple("SGD", ["lr", "momentum", "l2_decay"])
sgd_smoother = smoother.SGDSmoother(model=net, loss_fn =nn.NLLLoss(), optim_params=SGDparams(0.01, 0.00, 1e-2),
                                stopper=stopping.EpochStopper(1), log_interval=10)
prolongation_op = prolongation.PairwiseAggProlongation()
restriction_op = restriction.PairwiseAggRestriction(interpolator.PairwiseAggCoarsener)
epoch_stopper = stopping.EpochStopper(1)


# Build Multigrid Hierarchy Levels/Grids
num_levels = 3

FAS_levels = builder.build_vcycle_levels(num_levels=num_levels,  presmoother=sgd_smoother, postsmoother=sgd_smoother,
                                        prolongation_operator=prolongation_op,
                                        restriction_operator=restriction_op,
                                        coarsegrid_solver=sgd_smoother, stopper=epoch_stopper,
                                        loss_function=nn.NLLLoss())

mg_scheme = mg.FASVCycle(FAS_levels)
training_alg = trainer.MultigridTrainer(dataloader=dataloader.trainloader,
                                        verbose=True,
                                        log=True,
                                        save=False,
                                        load=False)


net.print('high')
#=====================================
# Train
#=====================================
print('Starting Training')
start = time.perf_counter()
trained_model = training_alg.train(model=net, multigrid=mg_scheme, cycles=1)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))
net.print('high')

#=====================================
# Test
#=====================================
evaluator = evaluator.CategoricalEvaluator()
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=dataloader.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
