"""
Holds Datasets
"""
import torch
import torchvision
import torchvision.transforms as transforms


class BaseDataLoader:
    # Top-level of datasets folder
    root = './datasets'
    num_workers = 0 # 4 * numGPUs
    batch_size = 64



class MnistData(BaseDataLoader):
    """
    Loads Mnist Dataset into Dataloaders
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), # convert to PyTorch Tensor
         transforms.Normalize((0.1307), (0.3081))]) # normalize with mean, standard deviation

    train_data = torchvision.datasets.MNIST(root=BaseDataLoader.root,
                                           train=True,
                                           download=True,
                                           target_transform=transform)
    test_data = torchvision.datasets.MNIST(root=BaseDataLoader.root,
                                          train=False,
                                          download=True,
                                          target_transform=transform)

    def __init__(self, batch_size=BaseDataLoader.batch_size):

        self.trainloader = torch.utils.data.DataLoader(MnistData.train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=self.num_workers)
        self.testloader = torch.utils.data.DataLoader(MnistData.test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=self.num_workers)


class Cifar10Data(BaseDataLoader):
    """
    Loads Cifar10 Dataset into Dataloaders
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root = './datasets', train = True,
                                            download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                              shuffle = True, num_workers = 2)

    testset = torchvision.datasets.CIFAR10(root = './datasets', train = False,
                                           download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
                                             shuffle = False, num_workers = 2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
