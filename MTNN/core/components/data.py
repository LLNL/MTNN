"""
Holds Pre-configured Dataloaders
"""
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as td


####################################################################
# Interface
###################################################################
class _BaseDataLoader:
    # Top-level of datasets folder
    root = './datasets'
    num_workers = 0 # 4 * numGPUs
    batch_size = 1


###################################################################
# Implementation
####################################################################
class FakeData(_BaseDataLoader):
   """
   Use this dataset for development and debugging
   Loads Pytorch Fake Dataset into Dataloaders
   https://pytorch.org/docs/stable/_modules/torchvision/datasets/fakedata.html#FakeData
   """
   preprocess = transforms.Compose(
       [transforms.ToTensor()])

   def __init__(self, imagesize:tuple, num_classes: int, trainbatch_size: int, testbatch_size:int):
       self.trainset = datasets.FakeData(size=trainbatch_size,
                                         image_size=imagesize,  # channels, width, height
                                         num_classes=num_classes,
                                         transform=FakeData.preprocess,
                                         target_transform=None,
                                         random_offset=0)
       self.testset = datasets.FakeData(size=testbatch_size,
                                        image_size=imagesize,
                                        num_classes=num_classes,
                                        transform=FakeData.preprocess,
                                        target_transform = None,
                                        random_offset=0)
       self.trainloader = td.DataLoader(self.trainset,
                                        batch_size=trainbatch_size,
                                        shuffle=False,
                                        num_workers = self.num_workers)
       self.testloader = td.DataLoader(self.testset,
                                        batch_size = testbatch_size,
                                        shuffle =False,
                                        num_workers = self.num_workers)




class MnistData(_BaseDataLoader):
    """
    Loads Pytorch Mnist Dataset into Dataloaders
    Image size is 28 x 28
        - 60,000 training images
        -10,000 testing images
    """
    preprocess = transforms.Compose(
        [transforms.ToTensor(), # Convert to 3 Channels
         transforms.Normalize((0.1307,), (0.3081,))])  # mean, standard deviation

    def __init__(self, trainbatch_size, testbatch_size):
        self.trainset = datasets.MNIST(root=_BaseDataLoader.root,
                                       train=True,
                                       download=True,
                                       transform=MnistData.preprocess)
        self.testset = datasets.MNIST(root=_BaseDataLoader.root,
                                      train=False,
                                      download=True,
                                      transform=MnistData.preprocess)
        self.trainloader = td.DataLoader(self.trainset,
                                                batch_size=trainbatch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers)
        self.testloader = td.DataLoader(self.testset,
                                               batch_size=testbatch_size,
                                               shuffle=True,
                                               num_workers=self.num_workers)



class CIFAR10Data(_BaseDataLoader):
    """
    Loads Cifar10 Dataset into Dataloaders
    """

    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, trainbatch_size, testbatch_size):
        self.trainset = datasets.CIFAR10(root = _BaseDataLoader.root,
                                         train = True,
                                         download = True,
                                         transform = CIFAR10Data.preprocess)
        self.testset = datasets.CIFAR10(root = _BaseDataLoader.root,
                                        train = False,
                                        download = True,
                                        transform = CIFAR10Data.preprocess)
        self.trainloader = td.DataLoader(self.trainset,
                                                batch_size = trainbatch_size,
                                                shuffle = True,
                                                num_workers = self.num_workers)
        self.testloader = td.DataLoader(self.testset,
                                               batch_size = testbatch_size,
                                               shuffle = True,
                                               num_workers = self.num_workers)



