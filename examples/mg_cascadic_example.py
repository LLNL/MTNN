# !/usr/bin/env/ python
"""
mg_cascadic_example.py
 * This file demonstrates the interface to the MTNN framework
 * Template script to run builder and train fully-connected neural network
 * with MNIST data using prolongation operators
 * with the MTNN core
"""
# standard
import time

# PyTorch
import torch
import torch.utils as utils
from torchvision import datasets, transforms


# local source
import MTNN.core as core


# TODO: Define logger


#####################################
# Load Data
#####################################

transform_fn = transforms.Compose(
    [transforms.Resize((28, 28)),
     transforms.ToTensor(),  # convert image to a PyTorch tensor
     transforms.Normalize((0.1307,), (0.3081,))])  # normalize with mean (tuple), standard deviation (tuple)


train_dataset = datasets.CIFAR10(root = './datasets',
                               train = True,
                               download = True,
                               transform = transform_fn)
trainloader = utils.data.DataLoader(train_dataset,
                                    batch_size = 10,
                                    shuffle = True,
                                    num_workers = 2)  # multi-process data loading
test_dataset = datasets.CIFAR10(root = './datasets',
                              train = False,
                              download = True,
                              transform = transform_fn)
test_loader = utils.data.DataLoader(test_dataset,
                                   batch_size = 10,
                                   shuffle = False,
                                   num_workers = 2)


#######################################
# Instantiate Model
#######################################
# 1. With
# 2. Use (pretrained) PyTorch Hub models
hub_model = torch.hub.load('pytorch/vision:v0.5.0', 'alexnet', pretrained=False )

# 3. With TorchVision

# 4. with Onnx file

# 5. MTNN Model class

#######################################
# Training
########################################
print('Starting Training')
start = time.perf_counter()
net = training_alg.train(mtnn_yamlmodel, TRAINLOADER)
stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))

#####################################
# Evaluate 
#####################################
tester = core.evaluator.BasicEvaluator()
print('Starting Testing')
start = time.perf_counter()
correct, total = tester.evaluate(net, dataset=TEST_DATASET)
stop = time.perf_counter()
print('Finished Testing (%.3f)' % (stop-start))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
"""
