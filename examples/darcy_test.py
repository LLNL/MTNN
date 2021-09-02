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
from MTNN.core.multigrid.operators import TauCorrector, smoother
import MTNN.core.multigrid.operators.SecondOrderRestriction as SOR
import MTNN.core.multigrid.operators.SecondOrderConverter as SOC
import MTNN.core.multigrid.operators.ParameterExtractor as PE
import MTNN.core.multigrid.operators.SimilarityMatcher as SimilarityMatcher
import MTNN.core.multigrid.operators.TransferOpsBuilder as TransferOpsBuilder
from MTNN.core.alg import trainer, evaluator
from MTNN.utils import deviceloader
import MTNN.core.multigrid.scheme as mg

# Darcy problem imports
darcy_path = "./datasets/darcy"
sys.path.append(darcy_path)
from PDEDataSet import *

def read_args(args):
    int_reader = lambda x : int(x)
    float_reader = lambda x : float(x)
    string_reader = lambda x : x
    bool_reader = lambda x : x.lower() in ("yes", "true", "t", "1")
    ensure_trailing_reader = lambda tr : lambda x : x.rstrip(tr) + tr
    array_reader = lambda element_reader : \
                   lambda x : [element_reader(z) for z in x.split(',')]

    # Define reader functions for each parameter                                                                                                                                                                                              
    reader_fns = { "num_cycles" : int_reader,
                   "num_levels": int_reader,
                   "smooth_iters": int_reader,
                   "conv_ch" : array_reader(int_reader),
                   "conv_kernel_width" : array_reader(int_reader),
                   "conv_stride" : array_reader(int_reader),
                   "fc_width" : array_reader(int_reader),
                   "loader_sizes" : array_reader(int_reader),
                   "momentum": float_reader,
                   "learning_rate": float_reader,
                   "weight_decay": float_reader,
                   "tau_corrector": string_reader,
                   "weighted_projection": bool_reader}

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

train_filename = darcy_path + '/train_data_32.npz'
test_filename = darcy_path + '/test_data_32.npz'
percent_train = 0.8
orig_filename = darcy_path + '/match_pde_data_u_Q_32_50000.npz'

if not path.exists(train_filename):
    print("Generating training and testing files.")
    input_data = np.load(orig_filename)
    nx = input_data['nx']
    ny = input_data['ny']
    x_coord = input_data['x']
    y_coord = input_data['y']
    u_input = input_data['u']
    Q_output = input_data['Q']
    training_set_size = int(percent_train * u_input.shape[0])

    np.savez(train_filename, nx=nx, ny=ny, x_coord=x_coord, y_coord=y_coord,
             u=u_input[:training_set_size, ], Q=Q_output[:training_set_size, :])
    np.savez(test_filename, nx=nx, ny=ny, x_coord=x_coord, y_coord=y_coord,
             u=u_input[training_set_size:, ], Q=Q_output[training_set_size:, :])

#define pytorch datset
print("Loading training and testing files.")
pde_dataset_train = PDEDataset(train_filename,transform=None, reshape=True)
pde_dataset_test = PDEDataset(test_filename,transform=None, reshape=True)
    

u0, Q0 = pde_dataset_train.__getitem__(index =0)

#next: https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html, part II

#perform dataloader
print('u shape at the first row : {}'.format(u0.size()))
print('u unsqueeze shape at the first row : {}'.format(u0.unsqueeze(0).size()))

BATCH_SIZE = 200
test_batch_size = 2000
train_loader = DataLoader(pde_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(pde_dataset_test, batch_size=test_batch_size, shuffle=True)

print("Train loader has size {}".format(len(train_loader)))

params = read_args(sys.argv)
print(params)

#=====================================
# Set up network architecture
#=====================================

conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
print("conv_info: ", conv_info)
net = models.ConvolutionalNet(conv_info, params["fc_width"] + [1], F.relu, lambda x : x)
#net = models.MultiLinearNet([1024, params["width"][0], params["width"][1], 1], F.relu, lambda x : x)


#=====================================
# Multigrid Hierarchy Components
#=====================================
class SGDparams:
    def __init__(self, lr, momentum, l2_decay):
        self.lr = lr
        self.momentum = momentum
        self.l2_decay = l2_decay

if params["tau_corrector"] == "null":
    tau = TauCorrector.NullTau
elif params["tau_corrector"] == "wholeset":
    tau = TauCorrector.WholeSetTau
elif params["tau_corrector"] == "minibatch":
    tau = TauCorrector.MinibatchTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = params["num_levels"]
FAS_levels = []
# Control number of pochs and learning rate per level
lr = params["learning_rate"] #0.01
momentum = params["momentum"] #float(sys.argv[3])
l2_decay = params["weight_decay"] #1.0e-4 #316
l2_scaling = [1.0, 1.0, 0.0]
smooth_pattern = [1, 1, 1, 8]
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

    converter = SOC.ConvolutionalConverter(net.num_conv_layers)
    parameter_extractor = PE.ParamMomentumExtractor(converter)
    gradient_extractor = PE.GradientExtractor(converter)
    matching_method = SimilarityMatcher.HEMCoarsener(similarity_calculator=SimilarityMatcher.StandardSimilarity(),
                                                     coarsen_on_layer=None)#[False, False, True, True])
    transfer_operator_builder = TransferOpsBuilder.PairwiseOpsBuilder(restriction_weighting_power=0.0, weighted_projection=params["weighted_projection"])
    restriction = SOR.SecondOrderRestriction(parameter_extractor, matching_method, transfer_operator_builder)
    prolongation = SOR.SecondOrderProlongation(parameter_extractor, restriction)
    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation, #prolongation_op(),
                      restriction = restriction, #restriction_op(interpolator.PairwiseAggCoarsener),
                      coarsegrid_solver = sgd_smoother,
                      num_epochs = smooth_pattern[level_idx],
                      corrector = tau(loss_fn, gradient_extractor))

    FAS_levels.append(aLevel)


class ValidationCallback:
    def __init__(self, val_dataloader, test_frequency = 1):
        self.val_dataloader = val_dataloader
        self.test_frequency = test_frequency
        self.best_seen_init = 100000.0
        self.best_seen = None
        self.best_seen_linf = None

    def __call__(self, levels, cycle):
        if (cycle + 1) % self.test_frequency != 0:
            return
        if self.best_seen is None:
            self.best_seen = [self.best_seen_init] * len(levels)
            self.best_seen_linf = [self.best_seen_init] * len(levels)
        for level in levels:
            level.net.eval()

        with torch.no_grad():
            total_test_loss = [0.0] * len(levels)
            test_linf_loss = [0.0] * len(levels)
            for mini_batch_data in self.val_dataloader:
                inputs, true_outputs  = deviceloader.load_data(mini_batch_data, levels[0].net.device)
                for level_ind, level in enumerate(levels):
                    outputs = level.net(inputs)
                    total_test_loss[level_ind] += level.presmoother.loss_fn(outputs, true_outputs)
                    linf_temp = torch.max (torch.max(torch.abs(true_outputs - outputs), dim=1).values)
                    test_linf_loss[level_ind] = max(linf_temp, test_linf_loss[level_ind])
                for level_ind in range(len(levels)):
                    if total_test_loss[level_ind] < self.best_seen[level_ind]:
                        self.best_seen[level_ind] = total_test_loss[level_ind]
                    if test_linf_loss[level_ind] < self.best_seen_linf[level_ind]:
                        self.best_seen_linf[level_ind] = test_linf_loss[level_ind]
                    print("Level {}: After {} cycles, validation loss is {}, best seen is {}, linf loss is {}, best seen linf is {}".format(level_ind, cycle, total_test_loss[level_ind], self.best_seen[level_ind], test_linf_loss[level_ind], self.best_seen_linf[level_ind]), flush=True)

        for level in levels:
            level.net.train()

num_cycles = params["num_cycles"] #int(sys.argv[2])
depth_selector = None #lambda x : 3 if x < 55 else len(FAS_levels)
mg_scheme = mg.VCycle(FAS_levels, cycles = num_cycles,
                      subsetloader = subsetloader.NextKLoader(params["smooth_iters"]),
                      depth_selector = depth_selector, 
                      validation_callback=ValidationCallback(((pde_dataset_test.u, pde_dataset_test.Q),), 1))
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
loss_fn = nn.MSELoss()
with torch.no_grad():
    for level in range(len(FAS_levels)):
        total_loss = 0.0
        num_samples = 0
        for batch_idx, mini_batch_data in enumerate(test_loader):
            input_data, target_data = deviceloader.load_data(mini_batch_data, net.device)
            outputs = FAS_levels[level].net(input_data)
            total_loss += loss_fn(target_data, outputs)
            num_samples += test_batch_size
        print('Level {}: Total and average loss on the test set are {}, {}'.format(level, total_loss, total_loss / num_samples))
stop = time.perf_counter()
print('Finished Testing (%.3fs)' % (stop-start))



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

