import time

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# system imports
import sys
sys.path.append("../../mtnnpython")
import matplotlib.pyplot as plt

# local
from MTNN.core.components import models, subsetloader
#from MTNN.core.multigrid.operators import *
from MTNN.core.multigrid.operators import tau_corrector, smoother
import core.multigrid.operators.second_order_transfer as SOR
import core.multigrid.operators.data_converter as SOC
import MTNN.core.multigrid.operators.similarity_matcher as SimilarityMatcher
import MTNN.core.multigrid.operators.transfer_ops_builder as TransferOpsBuilder
from MTNN.core.alg import trainer
import MTNN.core.multigrid.scheme as mg

# Example execution:
# python circle_test.py rand_seed=0 width=24 num_levels=3 learning_rate=0.1 momentum=0.9 weight_decay=1e-9 num_cycles=200
#

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
                   "weight_decay": float_reader,
                   "rand_seed": int_reader}

    params_dict = dict()
    try:
        for a in args[1:]:
            tokens = a.split('=')
            params_dict[tokens[0]] = reader_fns[tokens[0]](tokens[1])
    except Exception as e:
        exit(str(e) + "\n\nCommand line format: python generate_linsys_data.py num_rows=[int] "
             "num_agents=[int] data_directory=[dir] config_directory=[dir]")
    return params_dict

params = read_args(sys.argv)
print(params)

# For reproducibility
torch.manual_seed(params["rand_seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(params["rand_seed"])
torch.set_printoptions(precision=5)

#======================================
# Generate training and validation data
#======================================

### 2D functions

def generate_radial_inputs(num_samples, max_radius = 1.0):
    """ Generate 2D data in a radial fashion.
    Ideally use num_samples that is a square of an even number.
    """
    num_angles = 2 * np.sqrt(num_samples)
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / num_angles)
    num_distances = np.sqrt(num_samples) / 2
    dist_interval = float(max_radius) / num_distances
    distances = np.arange(dist_interval, max_radius + dist_interval, dist_interval).reshape([-1, 1])
    point_subsets = []
    for ang in angles:
        normalized_point = np.array([[np.sin(ang), np.cos(ang)]])
        point_subsets.append(distances @ normalized_point)
    return np.concatenate(point_subsets, axis=0)

def generate_circle_distance_data(num_samples = 100, slope = 1.0, stddev = 0.0):
    x = generate_radial_inputs(num_samples)
    y = np.max([np.zeros(num_samples), np.sqrt(np.sum(x*x, axis=1)) - 0.5], axis=0)
    y += np.random.normal(loc=0.0, scale=stddev, size=y.shape)
    return x, y

class Synthetic_Dataset(Dataset):
    def __init__(self, x_vals, y_vals):
        self.x = torch.from_numpy(x_vals.astype(np.float32)).reshape([-1, x_vals.shape[-1]])
        self.y = torch.from_numpy(y_vals.astype(np.float32)).reshape([-1, 1])

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index,:], self.y[index,:]

num_samples=400
stddev=0.0
x_train, y_train = generate_circle_distance_data(num_samples=num_samples, stddev=stddev)
num_test_samples = 900
x_test, y_test = generate_circle_distance_data(num_samples=num_test_samples, stddev=0.0)

batch_size = num_samples
train_dataset = Synthetic_Dataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = Synthetic_Dataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

#=======================================
# Set up network architecture
#=======================================

layer_widths = [2] + params["width"] + [1]
net = models.MultiLinearNet(layer_widths, F.relu, lambda x : x)

#======================================
# Training setup
#=====================================

class SGDparams:
    def __init__(self, lr, momentum, l2_decay):
        self.lr = lr
        self.momentum = momentum
        self.l2_decay = l2_decay
tau = tau_corrector.BasicTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = params["num_levels"]
FAS_levels = []
# Control number of pochs and learning rate per level
lr = params["learning_rate"] #0.01
momentum = params["momentum"] #float(sys.argv[3])
l2_decay = params["weight_decay"] #1.0e-4 #316
l2_scaling = [1.0, 1.0, 1.0, 1.0]
smooth_pattern = [181] * 4 #[128, 128, 128, 128]
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

    parameter_extractor = SOC.ParameterExtractor(SOC.MultiLinearConverter())
    matching_method = SimilarityMatcher.HEMCoarsener(similarity_calculator=SimilarityMatcher.StandardSimilarity())
    transfer_operator_builder = TransferOpsBuilder.PairwiseOpsBuilder(restriction_weighting_power=0.0, weighted_projection=False)
    restriction = SOR.SecondOrderRestriction(parameter_extractor, matching_method, transfer_operator_builder)
    prolongation = SOR.SecondOrderProlongation(parameter_extractor, restriction)
    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation,
                      restriction = restriction,
                      coarsegrid_solver = sgd_smoother,
                      num_epochs = smooth_pattern[level_idx],
                      corrector = tau(loss_fn))

    FAS_levels.append(aLevel)


num_cycles = params["num_cycles"]
depth_selector = None #lambda x : 3 if x < 55 else len(FAS_levels)
mg_scheme = mg.VCycle(FAS_levels, cycles = num_cycles,
                      subsetloader = subsetloader.WholeSetLoader(),
                      depth_selector = depth_selector)
mg_scheme.test_loader = test_loader
training_alg = trainer.MultigridTrainer(scheme=mg_scheme,
                                        verbose=True,
                                        log=True,
                                        save=False,
                                        load=False)

#==============================================
# Set bias to have reasonable inflection points
#==============================================
w = net.layers[0].weight.data
b = net.layers[0].bias.data
zp = torch.from_numpy(np.random.uniform(low=0.0, high=1.0, size=len(b)).astype(np.float32))
b[:] = torch.norm(w, dim=1) * zp

#============================================
# Useful 2D Plotting function
#============================================

def plot_outputs(net, coarse_net, loader):
    with torch.no_grad():
        w = net.layers[0].weight.data
        b = net.layers[0].bias.data
        print("Weights are ", w)
        print("Biases are ", b)
        print("L2 weights are ", net.layers[1].weight.data)
        print("Activation distances are ", -b / torch.norm(w, dim=1))

        if coarse_net:
            fig, axs = plt.subplots(3, 2)
        else:
            fig, axs = plt.subplots(2,2)
        color_levels = int(np.sqrt(num_test_samples) / 2) #[a/10.0 for a in range(10)]
        
        cntr00 = axs[0,0].tricontourf(x_test[:,0], x_test[:,1], y_test, levels=color_levels)
        fig.colorbar(cntr00, ax=axs[0,0])
        axs[0,0].plot(x_test[:,0], x_test[:,1], 'ko', ms=1)
        axs[0,0].set_title("True function")
        
        cntr01 = axs[0,1].tricontourf(x_train[:,0], x_train[:,1], y_train, levels=color_levels)
        fig.colorbar(cntr01, ax=axs[0,1])
        axs[0,1].plot(x_train[:,0], x_train[:,1], 'ko', ms=1)
        axs[0,1].set_title("Noisy training function")
        
        x, y_true = next(iter(loader))
        y_net = net(x)
        cntr10 = axs[1,0].tricontourf(x[:,0], x[:,1], y_net[:,0], levels=color_levels)
        fig.colorbar(cntr10, ax=axs[1,0])
        axs[1,0].plot(x[:,0], x[:,1], 'ko', ms=1)
        # Draw neuron arrows
        norm_w = torch.norm(w, dim=1)
        dir_w = w / norm_w.reshape([-1,1])
        start_dist = -b / norm_w
        for i in range(w.shape[0]):
            pos = dir_w[i,:] * start_dist[i]
            mycolor = 'b' if net.layers[1].weight.data[0,i] > 0 else 'r'
            axs[1,0].arrow(pos[0], pos[1], .1*dir_w[i,0], .1*dir_w[i,1], width=.003, head_width=.02, color=mycolor)
        axs[1,0].set_title("Neural net function")

        resid = y_net[:,0] - y_true[:,0]
        cntr11 = axs[1,1].tricontourf(x[:,0], x[:,1], resid, levels=int(np.sqrt(num_samples) / 2))
        fig.colorbar(cntr11, ax=axs[1,1])
        axs[1,1].plot(x[:,0], x[:,1], 'ko', ms=1)
        axs[1,1].set_title("Residual")

        if coarse_net:
            coarse_y_net = coarse_net(x)
            coarse_resid = coarse_y_net[:,0] - y_true[:,0]
            cntr20 = axs[2,0].tricontourf(x[:,0], x[:,1], coarse_resid, levels=int(np.sqrt(num_samples)/2))
            fig.colorbar(cntr20, ax=axs[2,0])
            axs[2,0].plot(x[:,0], x[:,1], 'ko', ms=1)
            axs[2,0].set_title("Coarse residual")
            
            cntr21 = axs[2,1].tricontourf(x[:,0], x[:,1], resid - coarse_resid, levels=int(np.sqrt(num_samples)/2))
            fig.colorbar(cntr21, ax=axs[2,1])
            axs[2,1].plot(x[:,0], x[:,1], 'ko', ms=1)
            axs[2,1].set_title("residual - Coarse_residual")
            print("Dot product: ", torch.sum(resid * coarse_resid))
            print("Cosine angle: ", torch.sum(resid * coarse_resid) / (torch.norm(resid) * torch.norm(coarse_resid)))

        plt.show()

def plot_outputs_1by1(net, coarse_net, loader):
    with torch.no_grad():
        w = net.layers[0].weight.data
        b = net.layers[0].bias.data
        print("Weights are ", w)
        print("Biases are ", b)
        print("L2 weights are ", net.layers[1].weight.data)
        print("Activation distances are ", -b / torch.norm(w, dim=1))

        color_levels = int(np.sqrt(num_test_samples) / 2) #[a/10.0 for a in range(10)]
        
        cntr00 = plt.tricontourf(x_test[:,0], x_test[:,1], y_test, levels=color_levels)
        plt.colorbar(cntr00)
        plt.plot(x_test[:,0], x_test[:,1], 'ko', ms=1)
        plt.title("True function")
        plt.show()
        
        cntr01 = plt.tricontourf(x_train[:,0], x_train[:,1], y_train, levels=color_levels)
        plt.colorbar(cntr01)
        plt.plot(x_train[:,0], x_train[:,1], 'ko', ms=1)
        plt.title("Noisy training function")
        plt.show()
        
        x, y_true = next(iter(loader))
        y_net = net(x)
        cntr10 = plt.tricontourf(x[:,0], x[:,1], y_net[:,0], levels=color_levels)
        plt.colorbar(cntr10)
        plt.plot(x[:,0], x[:,1], 'ko', ms=1)
        # Draw neuron arrows
        norm_w = torch.norm(w, dim=1)
        dir_w = w / norm_w.reshape([-1,1])
        start_dist = -b / norm_w
        for i in range(w.shape[0]):
            pos = dir_w[i,:] * start_dist[i]
            mycolor = 'y' if net.layers[1].weight.data[0,i] > 0 else 'r'
            plt.arrow(pos[0], pos[1], .1*dir_w[i,0], .1*dir_w[i,1], width=.003, head_width=.02, color=mycolor)
        plt.title("Neural net function")
        plt.show()

        resid = y_net[:,0] - y_true[:,0]
        cntr11 = plt.tricontourf(x[:,0], x[:,1], resid, levels=int(np.sqrt(num_samples) / 2))
        plt.colorbar(cntr11)
        plt.plot(x[:,0], x[:,1], 'ko', ms=1)
        plt.title("Residual")
        plt.show()

#=====================================
# Train
#=====================================

coarse_net = mg_scheme.levels[1].net if len(mg_scheme.levels) > 1 else None
plot_outputs(net, coarse_net, test_loader)

print('Starting Training')
start = time.perf_counter()
mg_scheme.stop_loss = 0.00
trained_model = training_alg.train(model=net, dataloader=train_loader)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop - start))


#===============================
# Test
#===============================
coarse_net = mg_scheme.levels[1].net if len(mg_scheme.levels) > 1 else None
plot_outputs(net, coarse_net, test_loader)
