"""
Holds Datasets
"""
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as torchdata


class BaseDataLoader:
    # Top-level of datasets folder
    root = './datasets'
    num_workers = 0 # 4 * numGPUs
    batch_size = 64


class MnistData(BaseDataLoader):
    """
    Loads Mnist Dataset into Dataloaders
    # TODO: Bug - transforms in DataLoader: TypeError: pic should be PIL Image or ndarray. Got <class 'int'>
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307), (0.3081))])  # mean, standard deviation

    def __init__(self, batch_size=BaseDataLoader.batch_size):
        self.trainset = datasets.MNIST(root=BaseDataLoader.root, train=True,
                                           download=True, target_transform=MnistData.transform)
        self.testset = datasets.MNIST(root = BaseDataLoader.root, train = False,
                                      download = True, target_transform = MnistData.transform)
        self.trainloader = torchdata.DataLoader(self.trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=self.num_workers)
        self.testloader = torchdata.DataLoader(self.testset, batch_size=batch_size,
                                               shuffle=True, num_workers=self.num_workers)


class CIFAR10Data(BaseDataLoader):
    
    # Loads Cifar10 Dataset into Dataloaders
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, batch_size=BaseDataLoader.batch_size):
        self.trainset = datasets.CIFAR10(root = BaseDataLoader.root, train = True,
                                         download = True, transform = CIFAR10Data.transform)
        self.testset = datasets.CIFAR10(root = BaseDataLoader.root, train = False,
                                        download = True, transform = CIFAR10Data.transform)
        self.trainloader = torchdata.DataLoader(self.trainset, batch_size = batch_size,
                                                shuffle = True, num_workers = self.num_workers)
        self.testloader = torchdata.DataLoader(self.testset, batch_size = batch_size,
                                               shuffle = True, num_workers = self.num_workers)


