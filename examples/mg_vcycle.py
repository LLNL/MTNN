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
from MTNN.core.multigrid.operators import *
from MTNN.core.alg import trainer, evaluator
import MTNN.core.multigrid.scheme as mg

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_printoptions(precision=5)


#=======================================
# Set-up
#=====================================
#test_data = data.TestData(trainbatch_size=10, testbatch_size=10)
fake_data = data.FakeData(imagesize=(1, 2, 2), num_classes=2, trainset_size=10, trainbatch_size= 1,
                          testset_size= 10, testbatch_size=10)
#vcycle_data = data.CycleLoader(fake_data.trainloader, alt=True)
net = models.MultiLinearNet([4, 3, 2], F.relu, F.log_softmax)

# With Mnist
#data = data.MnistData(trainbatch_size=100, testbatch_size=100)
#net = models.MultiLinearNet([784, 50, 25, 10], F.relu, F.log_softmax)

#=====================================
# Multigrid Hierarchy Components
#=====================================
SGDparams = namedtuple("SGDparams", ["lr", "momentum", "l2_decay"])
prolongation_op = prolongation.PairwiseAggProlongation
restriction_op = restriction.PairwiseAggRestriction
tau = tau_corrector.BasicTau
subsetloader_type = subsetloader.WholeSetLoader

# Build Multigrid Hierarchy Levels/Grids
num_levels = 3
FAS_levels = []
# Control number of pochs and learning rate per level
for level_idx in range(0, num_levels):
    if level_idx == 0:
        optim_params = SGDparams(lr=0.01, momentum=0.00, l2_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
    elif level_idx == 1:
        optim_params = SGDparams(lr=0.01, momentum=0.00, l2_decay=1e-2)
        loss_fn = nn.NLLLoss()
    else:
        optim_params = SGDparams(lr=0.01, momentum=0.00, l2_decay=1e-2)
        loss_fn = nn.NLLLoss()
    sgd_smoother = smoother.SGDSmoother(model = net, loss_fn = loss_fn,
                                        optim_params = optim_params,
                                        log_interval = 1)

    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation_op(),
                      restriction = restriction_op(interpolator.PairwiseAggCoarsener),
                      coarsegrid_solver = sgd_smoother,
                      corrector = tau(loss_fn))



    FAS_levels.append(aLevel)


mg_scheme = mg.VCycle(FAS_levels, cycles = 2, subsetloader = subsetloader_type())
training_alg = trainer.MultigridTrainer(scheme=mg_scheme,
                                        verbose=True,
                                        log=True,
                                        save=False,
                                        load=False)

#=====================================
# Train
#=====================================
print('Starting Training')
start = time.perf_counter()
trained_model = training_alg.train(model=net, dataloader=fake_data.trainloader)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))
#=====================================
# Test
#=====================================
evaluator = evaluator.CategoricalEvaluator()
print('Starting Testing')
start = time.perf_counter()
correct, total = evaluator.evaluate(model=net, dataloader=fake_data.testloader)
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
