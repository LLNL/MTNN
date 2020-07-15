"""
Script for debugging. To use with  FAS branch commit d628ddb1f92
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import numpy as np
import pdb

# MTNN stuff
import MTNN

import sys
sys.path.append("/Users/mao6/Workspace/Git/mtnnFAS.git/mtnnpython")
print(sys.path)

# Reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
torch.set_printoptions(precision=5)


batch_size=10

# Dataset
transforms = transforms.Compose([transforms.ToTensor()])
image_size = (1,2,2)
num_classes = 2
trainset = datasets.FakeData(size = 10,image_size = image_size,  # channels, width, height
                                     num_classes = num_classes,
                                     transform = transforms,
                                     target_transform = None,
                                     random_offset = 0)
testset = datasets.FakeData(size = 10,
                                    image_size = image_size,
                                    num_classes = num_classes,
                                    transform = transforms,
                                    target_transform = None,
                                    random_offset = 0)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False)




first_batch = next(iter(trainloader))
trainloader = [first_batch] #*100

activation = F.relu #torch.sigmoid
output_layer = F.log_softmax #F.softmax

net = MTNN.Net([4, 3, 2], activation=activation, output_layer=output_layer, weight_fill=1, bias_fill=1 )
for layers in net.layers:
    print(layers.weight)
    print(layers.bias)

#pdb.set_trace()

#
# Do training.
#

print("Creating the training algorithm")

#
#smoother0 = MTNN.TrainingAlgSmoother(alg=MTNN.SGDTrainingAlg(lr=0.01, momentum=0.0, doprint=False),
#                                     stopping=MTNN.SGDStoppingCriteria(num_epochs=10))
#obj_func0 = MTNN.ObjFunc(nn.NLLLoss(), l2_decay=1e-3)
#smoother0.smooth(net, trainloader, obj_func0, None, None)

num_levels  = 3
num_Vcycles = 1
#
FAS_levels = []
for i in range(0, num_levels):
   if i == 0:
      learning_rate_i = 0.01
      l2_decay_i = 1e-2
      num_epochs_i = 1
   elif i == 1:
      learning_rate_i = 0.01
      l2_decay_i = 1e-2
      num_epochs_i = 1
   else:
      learning_rate_i = 0.01
      l2_decay_i = 1e-2
      num_epochs_i = 1

   smoother = MTNN.TrainingAlgSmoother(alg=MTNN.SGDTrainingAlg(lr=learning_rate_i, momentum=0.0, doprint=False),
                                       stopping=MTNN.SGDStoppingCriteria(num_epochs=num_epochs_i))



   obj_func = MTNN.ObjFunc(nn.NLLLoss(), l2_decay=l2_decay_i)


   if (i < num_levels-1):
      agg = MTNN.AggInterpolator()
      hem = MTNN.HEMCoarsener(randseq=False)
      #presmoother, postsmoother, coarsener, prolongation, restriction, coarsesolver, level_id, objective_function
      #smoother, smoother, hem, agg, agg, None

      level = MTNN.Level(smoother, smoother, hem, agg, agg, None, i, obj_func)
   else:
      level = MTNN.Level(None, None, None, None, None, smoother, num_levels-1, obj_func)

   FAS_levels.append(level)

training_alg = MTNN.FASMG(FAS_levels)

print('Starting Training')
start = time.perf_counter()

for i in range(0, num_Vcycles):
   print('------ FAS cycle %d' % (i))
   #resetup = True
   resetup = i == 0
   net = training_alg.Vtrain(net, trainloader, resetup)

stop = time.perf_counter()
print('Finished Training (%.3fs)' % (stop-start))

#pdb.set_trace()

tester = MTNN.BasicEvaluator()

print('Starting Testing')
start = time.perf_counter()
correct, total = tester.evaluate(net, testloader)
stop = time.perf_counter()
print('Finished Testing (%.3f)' % (stop-start))
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
