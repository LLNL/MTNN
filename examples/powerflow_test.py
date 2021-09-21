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
from torch.utils.data import Dataset, DataLoader

# system imports
import sys
sys.path.append("../../mtnnpython")
from os import path

# local
from MTNN.core.components import data, models, subsetloader
from MTNN.core.multigrid.operators import *
import MTNN.core.multigrid.operators.coarsener as coarsener
from MTNN.core.alg import trainer, evaluator
from MTNN.utils import deviceloader
import MTNN.core.multigrid.scheme as mg

# Darcy problem imports
sys.path.append("./datasets/darcy")
from PDEDataSet import *

def read_args(args):
    int_reader = lambda x : int(x)
    float_reader = lambda x : float(x)
    string_reader = lambda x : x
    ensure_trailing_reader = lambda tr : lambda x : x.rstrip(tr) + tr
    array_reader = lambda element_reader : \
                   lambda x : [element_reader(z) for z in x.split(',')]

    # Define reader functions for each parameter                                                                                                                                                                                              
    reader_fns = { "num_cycles" : int_reader,
                   "num_levels": int_reader,
                   "width" : array_reader(int_reader),
                   "loader_sizes" : array_reader(int_reader),
                   "momentum": float_reader,
                   "learning_rate": float_reader,
                   "weight_decay": float_reader}

    params_dict = dict()
    try:
        for a in args[1:]:
            tokens = a.split('=')
            params_dict[tokens[0]] = reader_fns[tokens[0]](tokens[1])
    except Exception as e:
        exit(str(e) + "\n\nCommand line format: python generate_linsys_data.py num_rows=[int] "
             "num_agents=[int] data_directory=[dir] config_directory=[dir]")
    return params_dict


# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_printoptions(precision=5)


#=======================================
# Set up data
#=====================================

data_mean = 2.44
data_std = 1.11
val_mean = 0.0 #1.12
val_std = 1.0 #0.06

class Powerflow_Dataset(Dataset):
    def __init__(self, file_path):
        my_dict = np.load(file_path, allow_pickle=True)
        self.U = torch.from_numpy(my_dict.item()['U'].astype(np.float32))
        self.U = self.U - data_mean #torch.mean(self.U, dim=0)
        self.U = self.U / data_std #torch.std(self.U, dim=0)
        self.Q = torch.from_numpy(my_dict.item()['Q'].reshape([-1,32,32]).astype(np.float32))
        self.Q = self.Q[:,16,:]
        self.Q = self.Q - val_mean #torch.mean(self.Q, dim=0)
        self.Q = self.Q / val_std #torch.std(self.Q, dim=0)

    def __len__(self):
        return self.Q.shape[0]

    def __getitem__(self, index):
        X = self.U[index,:]
        y = self.Q[index,:]
        return X,y

train_filename = './datasets/powerflow/powerflow_resistance_train.npy'
test_filename = './datasets/powerflow/powerflow_resistance_test.npy'

train_dataset = Powerflow_Dataset(train_filename)
test_dataset = Powerflow_Dataset(test_filename)

BATCH_SIZE = 200
test_batch_size = 10000
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print(len(train_loader))

params = read_args(sys.argv)
print(params)

d = np.load(train_filename, allow_pickle=True)
U = d.item()['U']
U = torch.from_numpy(U.astype(np.float32))
U = U - data_mean #torch.mean(U, dim=0)
U = U / data_std #torch.std(U, dim=0)
U = U[:5000,:]
U = U.to("cuda:0")

#=====================================
# Set up network architecture
#=====================================

layer_widths = [1024] + params["width"] + [32]
net = models.MultiLinearNet(layer_widths, F.relu, lambda x : x)

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
num_levels = params["num_levels"]
FAS_levels = []
# Control number of pochs and learning rate per level
lr = params["learning_rate"] #0.01
momentum = params["momentum"] #float(sys.argv[3])
l2_decay = params["weight_decay"] #1.0e-4 #316
l2_scaling = [1.0, 1.0, 1.0, 1.0]
smooth_pattern = [1, 1, 1, 1]
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

    mycoarsener = coarsener.HEMCoarsener(similarity_calculator=coarsener.StandardSimilarity())
    aggregator = interpolator.PairwiseAggCoarsener(mycoarsener)
    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation_op(),
                      restriction = restriction_op(aggregator), #interpolator.PairwiseAggCoarsener),
                      coarsegrid_solver = sgd_smoother,
                      num_epochs = smooth_pattern[level_idx],
                      corrector = tau(loss_fn))

    FAS_levels.append(aLevel)


num_cycles = params["num_cycles"] #int(sys.argv[2])
depth_selector = None #lambda x : 3 if x < 55 else len(FAS_levels)
mg_scheme = mg.VCycle(FAS_levels, cycles = num_cycles,
                      subsetloader = subsetloader.CyclingNextKLoader(params["loader_sizes"]), #NextKLoader(4),
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
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))


#=====================================
# Test
#=====================================
print('Starting Testing')
start = time.perf_counter()
total_loss = 0.0
num_samples = 0
loss_fn = nn.MSELoss()
with torch.no_grad():
    for batch_idx, mini_batch_data in enumerate(test_loader):
        input_data, target_data = deviceloader.load_data(mini_batch_data, net.device)
        outputs = net(input_data)
        total_loss += loss_fn(target_data, outputs)
        num_samples += test_batch_size
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))
print('Total and average loss on the test set: {0}, {1}'.format(total_loss, total_loss / num_samples))



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

