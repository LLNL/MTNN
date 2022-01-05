from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class DarcyDataset(Dataset):

    def __init__(self, file_path, transform=None, reshape = True):
        self.data = np.load(file_path)
        if reshape:
            self.u = torch.from_numpy(self.data['u'].astype(np.float32).reshape(-1, 1, self.data['nx'], self.data['ny']))
        else:
            self.u = torch.from_numpy(self.data['u'].astype(np.float32))
        self.Q = torch.from_numpy(self.data['Q'].astype(np.float32))
        self.transform = transform

    def __len__(self):
        return len(self.data['Q'])

    def __getitem__(self, index):
        X = self.u[index,:]
        y = self.Q[index]

        if self.transform is not None:
            X = self.transform(X)

        return X,y

def get_loaders(percent_train, train_batch_size, darcy_path = "./datasets/darcy"):
    train_filename = darcy_path + '/train_data_32.npz'
    test_filename = darcy_path + '/test_data_32.npz'
    orig_filename = darcy_path + '/match_pde_data_u_Q_32_50000.npz'
    
    if not path.exists(train_filename) or not path.exists(test_filename):
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

    pde_dataset_train = DarcyDataset(train_filename,transform=None, reshape=True)
    pde_dataset_test = DarcyDataset(test_filename,transform=None, reshape=True)

    train_loader = DataLoader(pde_dataset_train, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(pde_dataset_test, batch_size=len(pde_dataset_test), shuffle=False)
    return train_loader, test_loader
