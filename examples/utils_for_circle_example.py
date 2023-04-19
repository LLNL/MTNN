import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from MTNN.utils import deviceloader

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



class CircleHelper:
    def __init__(self, num_train_samples, num_test_samples):
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.generate_data()
        
    def generate_data(self):
        stddev=0.0
        x_train, y_train = generate_circle_distance_data(num_samples=self.num_train_samples, stddev=stddev)
        self.x_train = x_train
        self.y_train = y_train
        x_test, y_test = generate_circle_distance_data(num_samples=self.num_test_samples, stddev=0.0)
        self.x_test = x_test
        self.y_test = y_test
        
        batch_size = self.num_train_samples
        self.train_dataset = Synthetic_Dataset(x_train, y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        
        self.test_dataset = Synthetic_Dataset(x_test, y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=len(x_test), shuffle=False)

    def get_dataloaders(self):
        return self.train_loader, self.test_loader
            
    def plot_outputs(self, net, level):
        with torch.no_grad():
            w = net.layers[0].weight.data.cpu()
            b = net.layers[0].bias.data.cpu()

            fig, axs = plt.subplots(2,2)
            color_levels = int(np.sqrt(self.num_test_samples) / 2)
            
            cntr00 = axs[0,0].tricontourf(self.x_test[:,0], self.x_test[:,1], self.y_test, levels=color_levels)
            fig.colorbar(cntr00, ax=axs[0,0])
            axs[0,0].plot(self.x_test[:,0], self.x_test[:,1], 'ko', ms=1)
            axs[0,0].set_title("True function")
            
            cntr01 = axs[0,1].tricontourf(self.x_train[:,0], self.x_train[:,1], self.y_train, levels=color_levels)
            fig.colorbar(cntr01, ax=axs[0,1])
            axs[0,1].plot(self.x_train[:,0], self.x_train[:,1], 'ko', ms=1)
            axs[0,1].set_title("Training function")
            
            x, y_true = next(iter(self.test_loader))
            y_net = net(x.to(deviceloader.get_device())).cpu()
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
            cntr11 = axs[1,1].tricontourf(x[:,0], x[:,1], resid, levels=int(np.sqrt(self.num_train_samples) / 2))
            fig.colorbar(cntr11, ax=axs[1,1])
            axs[1,1].plot(x[:,0], x[:,1], 'ko', ms=1)
            axs[1,1].set_title("Residual")

            fig.suptitle("Hierarchy Level {}".format(level))

            plt.show()
