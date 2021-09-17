"""
Example of FCNet SGD train
"""
import time
from collections import namedtuple

# PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb

# system imports
import sys
sys.path.append("../../mtnnpython")
from os import path

# local
from MTNN.core.components import data, models, subsetloader
from MTNN.core.multigrid.operators import *
from MTNN.core.alg import trainer, evaluator
from MTNN.utils import deviceloader
import MTNN.core.multigrid.scheme as mg

# Poisson problem imports
#sys.path.append("../data/darcy")
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

filename = '../data/Poisson4.npz'
data = np.load(filename)
Ndata = len(data['Kappa'])
Ntest = int(Ndata/10)
Ntrain = Ndata - Ntest
nx = 32

#pdb.set_trace()

perm = np.random.permutation(Ndata)

#define pytorch datset
print("Loading training and testing files.")
dataset_all = PDEDataset(data, select=None, transform=None, reshape=False, job=2)
dataset_train = PDEDataset(data, perm[0:Ntrain], transform=None, reshape=False, job=2)
dataset_test = PDEDataset(data, perm[Ntrain:Ndata], transform=None, reshape=False, job=2)

BATCH_SIZE = 200
train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)

class Net(nn.Module):
   def __init__(self):
      super(Net, self).__init__()
      # torch.nn.Flatten(start_dim=1, end_dim=-1)
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(3 * nx * nx, 400)
      self.fc2 = nn.Linear(400, 400)
      self.fc3 = nn.Linear(400, nx * nx)
      #self.fc3 = nn.Linear(400, 1)

   def forward(self, x):
      x = self.flatten(x)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      x = x.reshape(-1, nx, nx)
      return x

net = Net()

#net.load_state_dict(torch.load("poisson10000.pth"))
print(net)

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# create a loss function
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, output, target):
        loss = torch.sqrt(torch.sum((output-target)**2)) / torch.sqrt(torch.sum(target**2))
        return loss

#loss_fn = nn.MSELoss()
loss_fn = L2Loss()

# plot
def test_k(dataset=dataset_test, net=net, nx=nx, k=0, seeplot=False):
   with torch.no_grad():
      # show images
      data_k, target_k = dataset[k]
      kappa_k = data_k[0:nx,:]
      out_k = net(data_k.reshape(1,-1,nx))
      out_k = out_k.reshape(nx, nx)
      loss_k = loss_fn(out_k, target_k)
      out_k = out_k.detach().numpy()
      print("Loss of testset[%d] = %.8f" % (k, loss_k))
      target_k = target_k.detach().numpy()
      #
      if seeplot:
         plt.figure()
         c = plt.imshow(kappa_k)
         plt.colorbar(c)
         plt.figure()
         c = plt.imshow(target_k)
         plt.colorbar(c)
         plt.figure()
         c = plt.imshow(out_k)
         plt.colorbar(c)

   return data_k, target_k, out_k, loss_k

# train
def train(train_loader, net, loss_fn, optimizer, log_interval = 10):
   train_loss = 0
   Ntrain = len(train_loader.dataset)
   train_loss_each = np.zeros(len(train_loader))
   i = 0
   for batch_idx, (data, target) in enumerate(train_loader):
      #data, target = Variable(data), Variable(target)
      #
      net_out = net(data)
      loss = loss_fn(net_out, target)
      # back prop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      #
      train_loss += loss.data.item()
      train_loss_each[i] = loss.data.item()
      #
      if (batch_idx + 1) % log_interval == 0:
         print(" loss: %10.7f [%6d / %6d]" % (loss.data.item(), batch_idx * BATCH_SIZE, len(train_loader)*BATCH_SIZE))
      i += 1
   #
   train_loss /= Ntrain
   print(" Train error (size %d): Average loss %e" % (len(dataset_train), train_loss))
   return train_loss, train_loss_each


# test
def test(test_loader, net, loss_fn):
   test_loss = 0
   Ntest = len(test_loader.dataset)
   test_loss_each = np.zeros(len(test_loader))
   i = 0
   with torch.no_grad():
      for (data, target) in test_loader:
         #data, target = Variable(data), Variable(target)
         #
         net_out = net(data)
         loss = loss_fn(net_out, target).data.item()
         #
         test_loss_each[i] = loss
         test_loss += loss
         i += 1
   #
   test_loss /= Ntest
   print(" Test  error (size %d): Average loss %e" % (len(dataset_test), test_loss))
   return test_loss, test_loss_each

#
Nepochs = 100
for i in range(Nepochs):
   print("Epoch %d\n = = = = = = = = = = = = = = = = = = = = = = = = = = =" % (i+1))
   avg_train_loss, train_loss = train(train_loader, net, loss_fn, optimizer, log_interval=1000)
   avg_test_loss, test_loss = test(test_loader, net, loss_fn)

#torch.save(net.state_dict(), "poisson20000.pth")

sort_i = np.argsort(test_loss)
sort_loss = test_loss[sort_i]
pdb.set_trace()

data_k, target_k, out_k, loss_k = test_k(k=sort_i[-1], seeplot=True)
pdb.set_trace()

