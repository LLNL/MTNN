"""
Example of VCycle multigrid with MTNN
"""
import time
from collections import namedtuple

# PyTorch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# local
from MTNN.core.components import data, models
from MTNN.core.multigrid.operators import smoother, restriction, prolongation, interpolator
from MTNN.core.alg import trainer, evaluator, stopping
import MTNN.core.multigrid.scheme as mg
import MTNN.utils.builder as levels

# Set seed for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)

# Set-up
# Load Data and Model
dataloader = data.TestData(imagesize=(1, 28, 28), trainbatch_size=10, testbatch_size=10)
net = models.MultiLinearNet([784, 10, 5, 10], F.relu, F.log_softmax)

# Multigrid Hierarchy Components
optim_params = namedtuple("SGD", ["lr", "momentum", "l2_decay"])
# TODO: Epoch Power Control number of epochs per optimizer per level
# TODO: Control the learning rate per level
# TODO: Control the l2 decay per optimizer per level
smoother = smoother.SGDSmoother(model=net, loss=nn.NLLLoss(), optim_params=optim_params(1e-2, 0.01, 0.09),
                                log_interval=100)
prolongation_op = prolongation.PairwiseAggProlongation()
restriction_op = restriction.PairwiseAggRestriction()


stopping_measure = stopping.EpochStopper(1)
# TODO: Refactor Stopping measure
#stopping_measure = stopping.CycleStopper(epochs=1, cycles=1)


# Build Multigrid Hierarchy Levels/Grids
num_levels = 3
FAS_levels = levels.build_vcycle_levels(num_levels=num_levels,  presmoother=smoother, postsmoother=smoother,
                                        prolongation_operator=prolongation_op,
                                        restriction_operator=restriction_op,
                                        coarsegrid_solver=smoother, stopper=stopping_measure,
                                        loss_function=nn.NLLLoss())


mg_scheme = mg.FASVCycle(FAS_levels)
training_alg = trainer.MultigridTrainer(dataloader=dataloader.trainloader,
                                        verbose=True,
                                        log=False,
                                        save=False,
                                        load=False)


net.print('med')
# Train
print('Starting Training')
start = time.perf_counter()
trained_model = training_alg.train(model=net, multigrid=mg_scheme, cycles=1)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))
net.print('med')


# Test
evaluator = evaluator.CategoricalEvaluator()
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=dataloader.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
