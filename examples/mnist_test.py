"""
Example of cascadic multigrid with MTNN
"""
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
from MTNN.core.multigrid.operators import tau_corrector, smoother
import core.multigrid.operators.second_order_transfer as SOR
import core.multigrid.operators.data_converter as SOC
import MTNN.core.multigrid.operators.similarity_matcher as SimilarityMatcher
import MTNN.core.multigrid.operators.transfer_ops_builder as TransferOpsBuilder
from MTNN.core.alg import trainer
import MTNN.core.multigrid.scheme as mg
from MTNN.utils import deviceloader

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
                   "conv_ch" : array_reader(int_reader),
                   "conv_kernel_width" : array_reader(int_reader),
                   "conv_stride" : array_reader(int_reader),
                   "fc_width" : array_reader(int_reader),
                   "smooth_iters" : int_reader,
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
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
torch.set_printoptions(precision=5)

# Load Data and Model
dataset = data.MnistData(trainbatch_size=200, testbatch_size=10)
train_loader = dataset.trainloader
test_loader = dataset.testloader

# Get similarity data
# sim_loader = data.MnistData(trainbatch_size=5000, testbatch_size=10).trainloader
# U, q = next(iter(sim_loader))
# U = U.reshape([U.shape[0], -1])
# U = U - torch.mean(U, dim=0)
# normvec = torch.norm(U, dim=0)
# normvec[normvec == 0] = 1.0
# U = U / normvec
# U = U.to("cuda:0")
# print(U.shape, type(U))

params = read_args(sys.argv)
print(params)
print("Running on {}".format(deviceloader.get_device()))
#============================
# Set up network architecture
#============================

conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
print("conv_info: ", conv_info)
net = models.ConvolutionalNet(conv_info, params["fc_width"] + [10], F.relu, F.log_softmax)
#net = models.MultiLinearNet([784, params["width"][0], params["width"][1], 10], F.relu, F.log_softmax)

#=====================================
# Multigrid Hierarchy Components
#=====================================
class SGDparams:
    def __init__(self, lr, momentum, l2_decay):
        self.lr = lr
        self.momentum = momentum
        self.l2_decay = l2_decay
#SGDparams = namedtuple("SGDparams", ["lr", "momentum", "l2_decay"])
#prolongation_op = prolongation.PairwiseAggProlongation
#restriction_op = restriction.PairwiseAggRestriction
tau = tau_corrector.OneAtaTimeTau #BasicTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = params["num_levels"]
FAS_levels = []
# Control number of pochs and learning rate per level
lr = params["learning_rate"]
momentum = params["momentum"]
l2_decay = params["weight_decay"]
l2_scaling = [1.0, 1.0, 1.0, 1.0]
smooth_pattern = [1, 1, 1, 1]
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

    parameter_extractor = SOC.ParameterExtractor(SOC.ConvolutionalConverter(net.num_conv_layers))
    matching_method = SimilarityMatcher.HEMCoarsener(similarity_calculator=SimilarityMatcher.StandardSimilarity())
    transfer_operator_builder = TransferOpsBuilder.PairwiseOpsBuilder()
    restriction = SOR.SecondOrderRestriction(parameter_extractor, matching_method, transfer_operator_builder)
    prolongation = SOR.SecondOrderProlongation(parameter_extractor, restriction)
    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation, #prolongation_op(),
                      restriction = restriction, #restriction_op(aggregator), #interpolator.PairwiseAggCoarsener),
                      coarsegrid_solver = sgd_smoother,
                      num_epochs = smooth_pattern[level_idx],
                      corrector = tau(loss_fn))

    FAS_levels.append(aLevel)


num_cycles = params["num_cycles"]
depth_selector = None #lambda x : 3 if x < 55 else len(FAS_levels)
mg_scheme = mg.VCycle(FAS_levels, cycles = num_cycles,
                      subsetloader = subsetloader.NextKLoader(params["smooth_iters"]),
                      depth_selector = depth_selector)
mg_scheme.test_loader = test_loader
training_alg = trainer.MultigridTrainer(scheme=mg_scheme,
                                        verbose=True,
                                        log=True,
                                        save=False,
                                        load=False)


# with torch.no_grad():
#     total_test_loss = 0.0
#     loss_fn = nn.CrossEntropyLoss()
#     for mini_batch_data in test_loader:
#         inputs, true_outputs  = deviceloader.load_data(mini_batch_data, net.device)
#         outputs = net(inputs)
#         total_test_loss += loss_fn(outputs, true_outputs)
#     print("Level 0: Initial validation loss is {}".format(total_test_loss), flush=True)

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

