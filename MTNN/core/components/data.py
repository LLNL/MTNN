"""
Holds Datasets
"""
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as td


class BaseDataLoader:
    # Top-level of datasets folder
    root = './datasets'
    num_workers = 0 # 4 * numGPUs
    batch_size = 64

class MnistData(BaseDataLoader):
    """
    Loads Mnist Dataset into Dataloaders
    """


    preprocess = transforms.Compose(
        [transforms.ToTensor(), # Convert to 3 Channels
         transforms.Normalize((0.1307,), (0.3081,))])  # mean, standard deviation

    def __init__(self, trainbatch_size, testbatch_size):
        self.trainset = datasets.MNIST(root=BaseDataLoader.root,
                                       train=True,
                                       download=True,
                                       transform=MnistData.preprocess)
        self.testset = datasets.MNIST(root=BaseDataLoader.root,
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





class CIFAR10Data(BaseDataLoader):
    """
    Loads Cifar10 Dataset into Dataloaders
    """

    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, trainbatch_size, testbatch_size):
        self.trainset = datasets.CIFAR10(root = BaseDataLoader.root,
                                         train = True,
                                         download = True,
                                         transform = CIFAR10Data.preprocess)
        self.testset = datasets.CIFAR10(root = BaseDataLoader.root,
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

# TESTING ONLY
def gen_data(linear_function):
    training_data = []
    input_data = []
    for x in range(3):
        for y in range(3):
            training_datum = ((x, y), linear_function(x, y))
            input_data.append((x, y))
            training_data.append(training_datum)
    #return input_data, training_data
    return training_data

def tensorize_data(training_data: list):
    """Tensorizes data and load into a Pytorch DataLoader.
    Args:
        training_data: <list> of tuples (input, output)

    Returns:
        dataloader: <torch.utils.data.DataLoader>

    """
    # Load Data_z into Pytorch dataloader
    tensor_data = []
    for i, data in enumerate(training_data):
        XY, Z = iter(data)

        # Convert list to float tensor
        input = torch.tensor(XY, dtype = torch.float, requires_grad = True)
        Z = torch.FloatTensor([Z])

        tensor_data.append((input, Z))
    dataloader = torch.utils.data.DataLoader(tensor_data, shuffle = False, batch_size = 1)

    return dataloader


class TestData:
    def __init__(self, linear_function):
        self.trainset = gen_data(linear_function)
        self.trainloader = tensorize_data(self.trainset)



