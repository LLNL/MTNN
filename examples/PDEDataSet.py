"""
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb

class PDEDataset(Dataset):
   def __init__(self, data, select=None, transform=None, reshape=False, job=1):
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
      self.reshape = reshape
      if job == 2:
         #self.fluxnorm = np.sum(np.sqrt(self.FluxX * self.FluxX + self.FluxY * self.FluxY), axis=(1,2));
         self.fluxnorm = np.sum(self.FluxX * self.FluxX, axis=(1,2));

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
         #target = fluxnorm.reshape(1)
      if self.reshape:
         target = target.reshape(-1)
      sample = [data, target]
      #
      if self.transform:
         sample = self.transform(sample)

      return sample
