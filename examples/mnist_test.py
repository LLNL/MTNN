"""
Example of cascadic multigrid with MTNN
"""
from collections import namedtuple
import time

import sys
sys.path.append("../../mtnnpython")

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# local
from MTNN.core.components import data, models, subsetloader
from MTNN.core.multigrid.operators import *
from MTNN.core.alg import trainer, evaluator
import MTNN.core.multigrid.scheme as mg
from MTNN.utils import deviceloader

# For reproducibility                                                                                                                                                                                                                         
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.set_printoptions(precision=5)

# Load Data and Model
data = data.MnistData(trainbatch_size=200, testbatch_size=10)
train_loader = data.trainloader
test_loader = data.testloader

#============================
# Set up network architecture
#============================

net = models.MultiLinearNet([784, 4096, 2048, 10], F.relu, F.log_softmax)



#=====================================
# Multigrid Hierarchy Components
#=====================================
class SGDparams:
    def __init__(self, lr, momentum, l2_decay):
        self.lr = lr
        self.momentum = momentum
        self.l2_decay = l2_decay
#SGDparams = namedtuple("SGDparams", ["lr", "momentum", "l2_decay"])
prolongation_op = prolongation.PairwiseAggProlongation
restriction_op = restriction.PairwiseAggRestriction
tau = tau_corrector.OneAtaTimeTau #BasicTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = int(sys.argv[1])
FAS_levels = []
# Control number of pochs and learning rate per level
lr = 0.02154
momentum = float(sys.argv[3])
l2_decay = 3.16e-4 # 0.00001 #316
l2_scaling = [1.0, 1.0, 0.0]
smooth_pattern = [1, 1, 2, 8]
for level_idx in range(0, num_levels):
    if level_idx == 0:
        optim_params = SGDparams(lr=lr, momentum=momentum, l2_decay=l2_scaling[level_idx]*l2_decay)
        loss_fn = nn.CrossEntropyLoss()
    else:
        optim_params = SGDparams(lr=lr, momentum=momentum, l2_decay=l2_scaling[level_idx]*l2_decay)
        loss_fn = nn.CrossEntropyLoss()
    sgd_smoother = smoother.SGDSmoother(model = net, loss_fn = loss_fn,
                                        optim_params = optim_params,
                                        log_interval = 1)

    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation_op(),
                      restriction = restriction_op(interpolator.PairwiseAggCoarsener),
                      coarsegrid_solver = sgd_smoother,
                      num_epochs = smooth_pattern[level_idx],
                      corrector = tau(loss_fn))

    FAS_levels.append(aLevel)


num_cycles = int(sys.argv[2])
depth_selector = None #lambda x : 3 if x < 55 else len(FAS_levels)
mg_scheme = mg.VCycle(FAS_levels, cycles = num_cycles,
                      subsetloader = subsetloader.NextKLoader(4),
                      depth_selector = depth_selector)
mg_scheme.test_loader = test_loader
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
mg_scheme.stop_loss = 0.00
trained_model = training_alg.train(model=net, dataloader=train_loader)
# print("Dropping momentum, adding a hierarchy level.")
# for level in FAS_levels:
#     level.presmoother.optim_params.momentum = momentum * 0.85
#     level.postsmoother.optim_params.momentum = momentum * 0.85
#     level.coarsegrid_solver.optim_params.momentum = momentum * 0.85
# mg_scheme.depth_selector = lambda x : 3
# trained_model = training_alg.train(model=trained_model, dataloader=train_loader)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))


#=====================================
# Test
#=====================================
print('Starting Testing')
start = time.perf_counter()
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, mini_batch_data in enumerate(test_loader):
        images, labels = deviceloader.load_data(mini_batch_data, net.device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Accuracy of the network on the test images: {0}'.format(float(correct) / total))







# # Build Multigrid Hierarchy
# prolongation_operator = prolongation.LowerTriangleProlongation(expansion_factor=3)
# SGDparams = namedtuple("SGDparams", ["lr", "momentum", "l2_decay"])
# optim_params = SGDparams(lr=0.01, momentum=0.00, l2_decay=1e-2)
# smoother = smoother.SGDSmoother



# mg_levels = builder.build_uniform_levels(num_levels=3,
#                                         presmoother = smoother(model=net, loss_fn =nn.CrossEntropyLoss(),
#                                                              optim_params=optim_params, stopper=stopping_measure,
#                                                              log_interval=10000),
#                                         postsmoother = smoother(model=net, loss_fn =nn.CrossEntropyLoss(),
#                                                               optim_params=optim_params, stopper=stopping_measure,
#                                                               log_interval=10000),
#                                         prolongation_operator = prolongation_operator,
#                                         restriction_operator = None,
#                                         coarsegrid_solver=smoother(model=net, loss_fn =nn.CrossEntropyLoss(),
#                                                                    optim_params=optim_params, stopper=stopping_measure,
#                                                                    log_interval=10000),

#                                         )

# mg_scheme = mg.Cascadic(mg_levels)

# training_alg = trainer.MultigridTrainer(dataloader=data.trainloader,
#                                         verbose=True,
#                                         log=False,
#                                         save=False,
#                                         load=False)
# evaluator = evaluator.CategoricalEvaluator()



# # Train
# print('Starting Training')
# start = time.perf_counter()
# trained_model = training_alg.train(model=net, multigrid=mg_scheme, cycles=1)
# stop = time.perf_counter()
# print('Finished Training (%.3fs)' % (stop - start))

# trained_model.print('high')


# # Test
# print('Starting Testing')
# start = time.perf_counter()
# correct, total = evaluator.evaluate(model=net, dataloader=data.testloader)
# stop = time.perf_counter()
# print('Finished Testing (%.3fs)' % (stop-start))
# print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

