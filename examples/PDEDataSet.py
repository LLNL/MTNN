import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PDEDataset(Dataset):
   def __init__(self, data, select=None, transform=None, flatten=False, job=1):
      Kappa = data['Kappa'].astype(np.float32)
      F = data['F'].astype(np.float32)
      U = data['U'].astype(np.float32)
      X = data['X'].astype(np.float32)
      Y = data['Y'].astype(np.float32)
      FluxX = data['FluxX'].astype(np.float32)
      FluxY = data['FluxY'].astype(np.float32)
      self.job = job
      #
      if select is None:
         self.Kappa = Kappa
         self.F = F
         self.U = U
         self.FluxX = FluxX
         self.FluxY = FluxY
      else:
         self.Kappa = Kappa[select,:,:]
         self.F = F[select,:,:]
         self.U = U[select,:,:]
         self.FluxX = FluxX[select,:,:]
         self.FluxY = FluxY[select,:,:]
      #
      self.X = X
      self.Y = Y
      self.transform = transform
      self.flatten = flatten
      if job == 2:
         #self.fluxnorm = np.sum(np.sqrt(self.FluxX * self.FluxX + self.FluxY * self.FluxY), axis=(1,2));
         self.fluxnorm = np.sum(self.FluxX * self.FluxX, axis=(1, 2))

   def __len__(self):
      return len(self.Kappa)

   def __getitem__(self, index):
      Kappa = self.Kappa[index]
      U = self.U[index]
      F = self.F[index]
      X = self.X
      Y = self.Y
      FluxX = self.FluxX[index]
      #FluxY = self.FluxY
      if self.job == 2:
         fluxnorm = self.fluxnorm[index]
      #
      data = np.vstack((Kappa,X,Y))
      data = torch.from_numpy(data)
      if self.job == 1:
         target = torch.from_numpy(U)
      elif self.job == 2:
         target = torch.from_numpy(FluxX)
         #target = fluxnorm.flatten(1)
      if self.flatten:
         target = target.reshape(-1)
      sample = [data, target]
      #
      if self.transform:
         sample = self.transform(sample)

      return sample


def get_loaders(percent_train, train_batch_size, flatten, filename="./datasets/poisson_data/Poisson4.npz"):
   data = np.load(filename)
   Ndata = len(data['Kappa'])
   Ntest = int(percent_train * Ndata)
   Ntrain = Ndata - Ntest
   
   perm = np.random.permutation(Ndata)
   
   dataset_all = PDEDataset(data, select=None, transform=None, flatten=flatten, job=2)
   dataset_train = PDEDataset(data, perm[0:Ntrain], transform=None, flatten=flatten, job=2)
   dataset_test = PDEDataset(data, perm[Ntrain:Ndata], transform=None, flatten=flatten, job=2)
   
   train_loader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True)
   test_loader = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
   return train_loader, test_loader

