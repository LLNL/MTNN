# !/usr/bin/env/ python
"""
Test script to run build and train fully-connected neural network
with MNIST data using prolongation operators
"""

# standard library packages
import os
import sys
import datetime
import argparse
import time
import yaml
import inspect # For testing.

# Pytorch packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
from torchvision import datasets, transforms


# local source
import MTNN
from MTNN import model as mf




if __name__ == "__main__":
