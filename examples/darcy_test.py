"""
Example of FAS VCycle
"""
import time
from collections import namedtuple

# PyTorch
import torch.nn as nn
import torch.nn.functional as F

# system imports
import sys
sys.path.append("../../MTNNPython")
from os import path

# local
from MTNN.core.components import models, subsetloader
from MTNN.core.multigrid.operators import taucorrector, smoother
import MTNN.core.multigrid.operators.second_order_transfer as SOR
import MTNN.core.multigrid.operators.data_converter as SOC
import MTNN.core.multigrid.operators.paramextractor as PE
import MTNN.core.multigrid.operators.similarity_matcher as SimilarityMatcher
import MTNN.core.multigrid.operators.transfer_ops_builder as TransferOpsBuilder
from MTNN.core.alg import trainer
from MTNN.core.multigrid.level import Level
#from MTNN.utils import deviceloader
import MTNN.core.multigrid.scheme as mg

from MTNN.utils.ArgReader import ArgReader
from MTNN.utils.validation_callbacks import RealValidationCallback

# Darcy problem imports
darcy_path = "./datasets/darcy"
sys.path.append(darcy_path)
from DarcyPDEDataSet import *

arg_reader = ArgReader()
params = arg_reader.read_args(sys.argv)
print(params)

# For reproducibility
torch.manual_seed(params["rand_seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(params["rand_seed"])
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

#=====================================
# Set up network architecture
#=====================================

nn_is_cnn = "conv_ch" in params
if nn_is_cnn:
    print("Using a CNN")
    conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
    print("conv_info: ", conv_info)
    net = models.ConvolutionalNet(conv_info, params["fc_width"] + [1], F.relu, lambda x : x)
else:
    print("Using a FC network")
    net = models.MultiLinearNet([1024] + params["fc_width"] + [1], F.relu, lambda x : x)

#=====================================
# Multigrid Hierarchy Components
#=====================================
# class SGDparams:
#     def __init__(self, lr, momentum, l2_decay):
#         self.lr = lr
#         self.momentum = momentum
#         self.l2_decay = l2_decay
SGDparams = namedtuple('SGDparams', ['lr', 'momentum', 'l2_decay'])

if params["tau_corrector"] == "null":
    tau = taucorrector.NullTau
elif params["tau_corrector"] == "wholeset":
    tau = taucorrector.WholeSetTau
elif params["tau_corrector"] == "minibatch":
    tau = taucorrector.MinibatchTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = params["num_levels"]
neural_net_levels = []
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
    if nn_is_cnn:
        converter = SOC.ConvolutionalConverter(net.num_conv_layers)
    else:
        converter = SOC.MultiLinearConverter()
    parameter_extractor = PE.ParamMomentumExtractor(converter)
    gradient_extractor = PE.GradientExtractor(converter)
    matching_method = SimilarityMatcher.HEMCoarsener(similarity_calculator=SimilarityMatcher.StandardSimilarity(), coarsen_on_layer=None)#[False, False, True, True])
    transfer_operator_builder = TransferOpsBuilder.PairwiseOpsBuilder_MatrixFree(weighted_projection=params["weighted_projection"])
    restriction = SOR.SecondOrderRestriction(parameter_extractor, matching_method, transfer_operator_builder)
    prolongation = SOR.SecondOrderProlongation(parameter_extractor, restriction)
    aLevel = Level(id=level_idx,
                   presmoother = sgd_smoother,
                   postsmoother = sgd_smoother,
                   prolongation = prolongation, #prolongation_op(),
                   restriction = restriction, #restriction_op(interpolator.PairwiseAggCoarsener),
                   coarsegrid_solver = sgd_smoother,
                   num_epochs = smooth_pattern[level_idx],
                   corrector = tau(loss_fn, gradient_extractor))

    neural_net_levels.append(aLevel)



num_cycles = params["num_cycles"]
depth_selector = None #lambda x : 3 if x < 55 else len(neural_net_levels)
val_dataloader = ((pde_dataset_test.u, pde_dataset_test.Q),)
mg_scheme = mg.VCycle(neural_net_levels, cycles = num_cycles,
                      subsetloader = subsetloader.NextKLoader(params["smooth_iters"]),
                      depth_selector = depth_selector, 
                      validation_callback=RealValidationCallback(val_dataloader, params["num_levels"], 1))
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
validation_callback = RealValidationCallback(val_dataloader, params["num_levels"])
validation_callback(neural_net_levels, "complete")

# start = time.perf_counter()
# loss_fn = nn.MSELoss()
# with torch.no_grad():
#     for level in range(len(neural_net_levels)):
#         total_loss = 0.0
#         num_samples = 0
#         for batch_idx, mini_batch_data in enumerate(test_loader):
#             input_data, target_data = deviceloader.load_data(mini_batch_data, net.device)
#             outputs = neural_net_levels[level].net(input_data)
#             total_loss += loss_fn(target_data, outputs)
#             num_samples += test_batch_size
#         print('Level {}: Total and average loss on the test set are {}, {}'.format(level, total_loss, total_loss / num_samples))
# stop = time.perf_counter()
# print('Finished Testing (%.3fs)' % (stop-start))
