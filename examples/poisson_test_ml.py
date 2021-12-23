"""
Example of FAS VCycle
"""

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# system imports
import sys
sys.path.append("../")

# local
from MTNN.core.components import models, subsetloader
from MTNN.core.multigrid.operators import taucorrector, smoother
import MTNN.core.multigrid.operators.second_order_transfer as SOR
import MTNN.core.multigrid.operators.data_converter as SOC
import MTNN.core.multigrid.operators.paramextractor as PE
import MTNN.core.multigrid.operators.similarity_matcher as SimilarityMatcher
import MTNN.core.multigrid.operators.transfer_ops_builder as TransferOpsBuilder
from MTNN.core.alg import trainer
from MTNN.utils import deviceloader
import MTNN.core.multigrid.scheme as mg

# Darcy problem imports
sys.path.append("../data")
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
                   "weighted_projection": bool_reader,
                   "data_filename" : string_reader,
                   "rand_seed" : bool_reader}

    params_dict = dict(fc_width=[400,500], num_levels=2, smooth_iters=4, learning_rate=0.01, momentum=0.9, weight_decay=0.0, num_cycles=1, tau_corrector="wholeset", weighted_projection=True, data_filename='/usr/workspace/mtnn/poisson_data/Poisson4.npz', rand_seed=0)
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

#=======================================
# Set up data
#=====================================

filename = params["data_filename"]#'/usr/workspace/mtnn/poisson_data/Poisson4.npz'
data = np.load(filename)
Ndata = len(data['Kappa'])
Ntest = int(Ndata/10)
Ntrain = Ndata - Ntest
nx = 32

perm = np.random.permutation(Ndata)

#define pytorch datset
print("Loading training and testing files.")
pde_dataset_train = PDEDataset(data, perm[0:Ntrain], transform=None, reshape=True, job=1)
pde_dataset_test = PDEDataset(data, perm[Ntrain:Ndata], transform=None, reshape=True, job=1)

#next: https://pytorch.org/tutorials/recipes/recipes/custom_dataset_transforms_loader.html, part II

#perform dataloader
#print('u shape at the first row : {}'.format(u0.size()))
#print('u unsqueeze shape at the first row : {}'.format(u0.unsqueeze(0).size()))

BATCH_SIZE = 200
test_batch_size = Ntest
train_loader = DataLoader(pde_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(pde_dataset_test, batch_size=test_batch_size, shuffle=False)


#=====================================
# Set up network architecture
#=====================================

nn_is_cnn = "conv_ch" in params
if nn_is_cnn:
    print("Using a CNN")
    conv_info = [x for x in zip(params["conv_ch"], params["conv_kernel_width"], params["conv_stride"])]
    print("conv_info: ", conv_info)
    net = models.ConvolutionalNet(conv_info, params["fc_width"] + [nx*nx], F.relu, lambda x : x)
else:
    print("Using a FC network")
    net = models.MultiLinearNet([3*nx*nx] + params["fc_width"] + [nx*nx], F.relu, lambda x : x)
#net = models.MultiLinearNet([3*nx*nx, params["width"][0], params["width"][1], nx*nx], F.relu, lambda x : x)
#net = models.MultiLinearNet([3*nx*nx, params["width"][0], params["width"][1], 1], F.relu, lambda x : x)

print(net)

# create a loss function
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, target):
        loss = torch.sqrt(torch.sum((output-target)**2)) / torch.sqrt(torch.sum(target**2))
        return loss

#=====================================
# Multigrid Hierarchy Components
#=====================================
class SGDparams:
    def __init__(self, lr, momentum, l2_decay):
        self.lr = lr
        self.momentum = momentum
        self.l2_decay = l2_decay

if params["tau_corrector"] == "null":
    tau = taucorrector.NullTau
elif params["tau_corrector"] == "wholeset":
    tau = taucorrector.WholeSetTau
elif params["tau_corrector"] == "minibatch":
    tau = taucorrector.MinibatchTau

# Build Multigrid Hierarchy Levels/Grids
num_levels = params["num_levels"]
FAS_levels = []
# Control number of epochs and learning rate per level
lr = params["learning_rate"] #0.01
momentum = params["momentum"] #float(sys.argv[3])
l2_decay = params["weight_decay"]*10 #1.0e-4 #316
l2_scaling = [0.0]*10
l2_scaling[0] = 1.0
l2_scaling[1] = 1.0
#smooth_pattern = [1, 2, 4, 8]
smooth_pattern = [1, 1, 1, 1]

loss_fn = L2Loss()

for level_idx in range(0, num_levels):
    if level_idx == 0:
        optim_params = SGDparams(lr=lr, momentum=momentum, l2_decay=l2_decay)
    else:
        optim_params = SGDparams(lr=lr, momentum=momentum, l2_decay=l2_scaling[level_idx]*l2_decay)
    sgd_smoother = smoother.SGDSmoother(model = net, loss_fn = loss_fn,
                                        optim_params = optim_params,
                                        log_interval = 1) #10 * BATCH_SIZE

    converter = SOC.ConvolutionalConverter(net.num_conv_layers)
    if nn_is_cnn:
        converter = SOC.ConvolutionalConverter(net.num_conv_layers)
    else:
        converter = SOC.MultiLinearConverter()
    parameter_extractor = PE.ParamMomentumExtractor(converter)
    gradient_extractor = PE.GradientExtractor(converter)
    matching_method = SimilarityMatcher.HEMCoarsener(similarity_calculator=SimilarityMatcher.StandardSimilarity(),
                                                     coarsen_on_layer=None)#[False, False, True, True])
    transfer_operator_builder = TransferOpsBuilder.PairwiseOpsBuilder_MatrixFree(weighted_projection=params["weighted_projection"])
    restriction = SOR.SecondOrderRestriction(parameter_extractor, matching_method, transfer_operator_builder)
    prolongation = SOR.SecondOrderProlongation(parameter_extractor, restriction)

    aLevel = mg.Level(id=level_idx,
                      presmoother = sgd_smoother,
                      postsmoother = sgd_smoother,
                      prolongation = prolongation,
                      restriction = restriction,
                      coarsegrid_solver = sgd_smoother,
                      num_epochs = smooth_pattern[level_idx],
                      corrector = tau(loss_fn, gradient_extractor)) # None

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
                    test_linf_loss[level_ind] += max(linf_temp, test_linf_loss[level_ind])
                for level_ind in range(len(levels)):
                    if total_test_loss[level_ind] < self.best_seen[level_ind]:
                        self.best_seen[level_ind] = total_test_loss[level_ind]
                    if test_linf_loss[level_ind] < self.best_seen_linf[level_ind]:
                        self.best_seen_linf[level_ind] = test_linf_loss[level_ind]
                    #print("Level {}: After {} cycles, validation loss is {}, best seen is {}, linf loss is {}, best seen linf is {}".format(level_ind, cycle, total_test_loss[level_ind], self.best_seen[level_ind], test_linf_loss[level_ind], self.best_seen_linf[level_ind]), flush=True)
            print("Level {} Ntest {}: After {} cycles, validation loss is {}, best seen is {}, linf loss is {}, best seen linf is {}".format(0, len(self.val_dataloader), cycle, total_test_loss[0], self.best_seen[0], test_linf_loss[0], self.best_seen_linf[0]), flush=True)

        for level in levels:
            level.net.train()

##
num_cycles = params["num_cycles"] #int(sys.argv[2])
depth_selector = None #lambda x : 3 if x < 55 else len(FAS_levels)
mg_scheme = mg.VCycle(FAS_levels, cycles = num_cycles, #1
                      subsetloader = subsetloader.NextKLoader(params["smooth_iters"]), # subsetloader.WholeSetLoader()
                      depth_selector = depth_selector,
                      validation_callback=ValidationCallback(test_loader, 1))
training_alg = trainer.MultigridTrainer(scheme=mg_scheme,
                                        verbose=True,
                                        log=True,
                                        save=False,
                                        load=False)

#=====================================
# Train
#=====================================
#print('Starting Training')
#start = time.perf_counter()
mg_scheme.stop_loss = 0.00
trained_model = training_alg.train(model=net, dataloader=train_loader)
# print("Dropping learning rate")
# for level in FAS_levels:
#     level.presmoother.optim_params.lr = lr / 10.0
#     level.postsmoother.optim_params.lr = lr / 10.0
#     level.coarsegrid_solver.optim_params.lr = lr / 10.0
# #mg_scheme.depth_selector = lambda x : 4
# trained_model = training_alg.train(model=trained_model, dataloader=train_loader)
#stop = time.perf_counter()
#print('Finished Training (%.3fs)' % (stop - start))

#=====================================
# Test
#=====================================
#print('Starting Testing')
#start = time.perf_counter()
total_loss = 0.0
Ntest = len(test_loader)
test_loss = np.zeros(Ntest)
i = 0
num_samples = 0
with torch.no_grad():
    for batch_idx, mini_batch_data in enumerate(test_loader):
        input_data, target_data = deviceloader.load_data(mini_batch_data, net.device)
        outputs = net(input_data)
        loss = loss_fn(outputs, target_data).data.item()
        total_loss += loss
        test_loss[i] = loss
        num_samples += test_batch_size
        i += 1
#stop = time.perf_counter()
#print('Finished Testing (%.3fs)' % (stop-start))
print(" Test error (size %d): Average loss %e" % (num_samples, total_loss / num_samples))


#=====================================
# done
#=====================================

# plot
def test_k(dataset=pde_dataset_test, net=net, nx=nx, k=0, seeplot=False):
    # show images
    data_k, target_k = dataset[k]
    data_k = data_k.to(net.device)
    target_k = target_k.to(net.device)
    out_k = net(data_k.reshape(1,-1,nx))
    loss_k = loss_fn(out_k, target_k)
    out_k = out_k.detach().cpu().numpy()
    print("Loss of testset[%d] = %.8f" % (k, loss_k))
    target_k = target_k.detach().cpu().numpy()
    #
    out_k = out_k.reshape(nx, nx)
    target_k = target_k.reshape(nx, nx)
    #
    if seeplot:
       plt.figure()
       c = plt.imshow(target_k)
       plt.colorbar(c)
       plt.figure()
       c = plt.imshow(out_k)
       plt.colorbar(c)

    return data_k, target_k, out_k, loss_k


sort_i = np.argsort(test_loss)
sort_loss = test_loss[sort_i]
test_k(k=sort_i[0],seeplot=True)
pdb.set_trace()

