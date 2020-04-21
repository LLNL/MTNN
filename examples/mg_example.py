# This file demonstrates the interface to the MTNN framework
#
# The classifier is the toy problem at:
#   https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
#
#

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time

# MTNN stuff
import MTNN

#
# Do data stuff
#

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

#
# Define the model architecture
#

print("Setting up the network.")

# This is 1/4th the size of the reference network
class Net(nn.Module):
    """A basic image classifier."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1_outchan = 6
        self.conv2_outchan = 8
        self.conv1 = nn.Conv2d(3, self.conv1_outchan, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.conv1_outchan, self.conv2_outchan, 5)
        self.fc1 = nn.Linear(self.conv2_outchan * 5 * 5, 30)
        self.fc2 = nn.Linear(30, 21)
        self.fc3 = nn.Linear(21, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv2_outchan * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#
# Now define the loss function and the per-level optimization strategy.
#
# Note that these will be propagated to each level in the
# hierarchy. At time of writing, this is a deep-copy, but over time we
# should add per-level specification of these properties.
#

loss = nn.CrossEntropyLoss()

#
# Do training.
#

print("Creating the training algorithm")
smoother=MTNN.TrainingAlgSmoother(
    alg=MTNN.SGDTrainingAlg(lr=0.001, momentum=0.9, doprint=True),
    stopping=MTNN.SGDStoppingCriteria(num_epochs=1))

interp=MTNN.IdentityInterpolator()

# Another example might be:
#interp = MTNN.LowerTriangular(refinement_factor=2)

training_alg = MTNN.CascadicMG(smoother=smoother, refinement=interp, num_levels=3)
stopping = None; # Cascadic MG is a "one-shot solver". The input is a
                 # coarse model and the output is a fine model.

print('Starting Training')
start = time.perf_counter()
net = training_alg.train(net, trainloader, loss, stopping)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))

#
# Test the output trained model
#

tester = MTNN.BasicEvaluator()

print('Starting Testing')
start = time.perf_counter()
correct, total = tester.evaluate(net, testloader)
stop = time.perf_counter()
print('Finished Testing (%.3f)' % (stop-start))
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
