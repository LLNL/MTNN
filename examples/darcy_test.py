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

# system imports
import sys
sys.path.append("../../mtnnpython")
from os import path

# local
from MTNN.core.components import data, models, subsetloader
from MTNN.core.multigrid.operators import *
from MTNN.core.alg import trainer, evaluator
import MTNN.core.multigrid.scheme as mg

# Darcy problem imports
sys.path.append("./data_darcy_multilevel")
from PDEDataSet import *

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_printoptions(precision=5)


#=======================================
# Set up data
#=====================================

train_filename = 'data_darcy_multilevel/train_data_32.npz'
test_filename = 'data_darcy_multilevel/test_data_32.npz'
percent_train = 0.8
orig_filename = 'data_darcy_multilevel/match_pde_data_u_Q_32_50000.npz'

if not path.exists(train_filename):
    print("Generating training and testing files.")
    input_data = np.load(orig_filename)
    nx = input_data['nx']
    ny = input_data['ny']
    x_coord = input_data['x']
    y_coord = input_data['y']
    u_input = input_data['u']
    Q_output = input_data['Q']
    training_set_size = int(percent_train * u_input.shape[-1])

    np.savez(train_filename, nx=nx, ny=ny, x_coord=x_coord, y_coord=y_coord,
             u=u_input[:training_set_size, ], Q=Q_output[:training_set_size, :])
    np.savez(test_filename, nx=nx, ny=ny, x_coord=x_coord, y_coord=y_coord,
             u=u_input[training_set_size:, ], Q=Q_output[training_set_size:, :])

#define pytorch datset
print("Loading training and testing files.")
pde_dataset_train = PDEDataset(train_filename,transform=None, reshape=False)
pde_dataset_test = PDEDataset(test_filename,transform=None, reshape=False)
    

u0, Q0 = pde_dataset_train.__getitem__(index =0)

#next: https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html, part II

#perform dataloader
print('u shape at the first row : {}'.format(u0.size()))
print('u unsqueeze shape at the first row : {}'.format(u0.unsqueeze(0).size()))

BATCH_SIZE = 200
test_batch_size = 2000
train_loader = DataLoader(pde_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(pde_dataset_test, batch_size=test_batch_size, shuffle=True)

print(len(train_loader))

#=====================================
# Set up network architecture
#=====================================

net = models.MultiLinearNet([1024, 2048, 1024, 1], F.relu, lambda x : x)


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
tau = tau_corrector.BasicTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = int(sys.argv[1])
FAS_levels = []
# Control number of pochs and learning rate per level
lr = 0.01
momentum = float(sys.argv[3])
l2_decay = 0.0001 #316
l2_scaling = [1.0, 1.0, 0.0]
smooth_pattern = [1, 2, 4, 8]
for level_idx in range(0, num_levels):
    if level_idx == 0:
        optim_params = SGDparams(lr=lr, momentum=momentum, l2_decay=l2_decay)
        loss_fn = nn.MSELoss()
    else:
        optim_params = SGDparams(lr=lr, momentum=momentum, l2_decay=l2_scaling[level_idx]*l2_decay)
        loss_fn = nn.MSELoss()
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
                      subsetloader = subsetloader.WholeSetLoader(), #NextKLoader(4),
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
# print("Dropping learning rate")
# for level in FAS_levels:
#     level.presmoother.optim_params.lr = lr / 10.0
#     level.postsmoother.optim_params.lr = lr / 10.0
#     level.coarsegrid_solver.optim_params.lr = lr / 10.0
# #mg_scheme.depth_selector = lambda x : 4
# trained_model = training_alg.train(model=trained_model, dataloader=train_loader)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))


#=====================================
# Test
#=====================================
print('Starting Testing')
start = time.perf_counter()
total_err = 0.0
num_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        true_Q= np.asarray(labels)
        outputs = net(images)
        dnn_Q = np.asarray(outputs)
        err_Q = np.abs(true_Q-dnn_Q)

        total_err += np.sum(err_Q)
        num_samples += test_batch_size
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Total and average error on the test set: {0}, {1}'.format(total_err, total_err / num_samples))



# 3 3936: Total and average error on the test set: 385.48570251464844, 0.038548570251464846

# 2 5117: Total and average error on the test set: 388.1219253540039, 0.03881219253540039

# 1 12792: Total and average error on the test set: 404.01805114746094, 0.040401805114746094


# stop at 0.45 training loss:
# 3 levels, After 2100 cycles, training loss is 0.43540745973587036
# 2100 * 3.25 WU = 6825.0 WU

# 2 levels, After 3500 cycles, training loss is 0.4409220516681671
# 3500 * 2.5 WU = 8750.0 WU

# 1 level, After 10500 cycles, training loss is 0.44982290267944336
# 105000 WU

