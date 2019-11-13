# !/usr/bin/env/ python

#%%
# Pytorch packages
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import MTNN
from MTNN import randomperturb


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # By default, bias is true
        self.fc1 = nn.Linear(2, 1) # row, column # input, output
        #self.fc2 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return x

def gen_data():
    data = []
    z = lambda x,y: 3 * x + 2 *y 
    for i in range(10):

        input = ((i,i), z(i,i))
        data.append(input)
    return data



#%%
# Using Simple net
net = Net()
print(net)
print(list(net.parameters()))

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.5)


#%%
# Creating dummy data for training
z = lambda x,y: 3 * x + 2 *y
data_z = gen_data()
print(data_z)


#%%
# Run the main training loop
#criterion = F.mse_loss()

for epoch in range(20):
    for i, data2 in enumerate(data_z):
        # unpack
        XY, Z = iter(data2)
       # print(X, Y, Z)
        # Tensorize input and wrap in Variable to apply gradient descent
        # requires_grad is False by default

        # Convert list to float tensor
        input = Variable(torch.FloatTensor(XY), requires_grad = True) #torch.Floattensor expects a list
        # For output, set requires_grad to False
        Z = Variable(torch.FloatTensor([Z]), requires_grad = False)
        # print("Input", input, input.size(),"output", Z)
        # Zero out previous gradients
        optimizer.zero_grad()
        outputs = net(input)  # outputs is predicted Y
        # print("OUTPUT", outputs, outputs.size())
        #print("WEIGHTS", net.fc1.weight)

        loss = F.mse_loss(outputs, Z)
        #print("LOSS", loss)


        # Backward should only be called on a scalar
        loss.backward() # loss.backward() equivalent to loss.backward(torch.Tensor([1]))
        # only valid if tensor contains a single element
        #print("GRADIENTS", net.fc1.weight.grad)
        # Update weights
        optimizer.step()
        if i % 1 == 0: # Check me please
            print("Epoch {} - loss: {}".format(epoch, loss.item ()))

#%%
# Check if network got to 3x+b
print("Trained weights")
print(list(net.parameters()))

# Test prediction from network
print("Prediction")
prediction = net(torch.ones(2, 2)) #rows, columns
print(prediction, prediction.size())


#%%

# Using the MTNN
model_config = yaml.load(open("/Users/mao6/proj/mtnnpython/MTNN/tests/test.yaml", "r"), Loader = yaml.SafeLoader)
model = MTNN.Model()
model.set_config(model_config)
print(model)
print(model.view_parameters())

#%%
print("Testing on MTNN Model")
# Testing on MTNN Model
# Load Data_z into dataloader
tensor_data_z = []
for i, data2 in enumerate(data_z):
    XY, Z = iter(data2)
    # Convert list to float tensor
    input = Variable(torch.FloatTensor(XY), requires_grad = True) #torch.Floattensor expects a list
    Z = Variable(torch.FloatTensor([Z]), requires_grad = False)
    tensor_data_z.append((input, Z))

print("Data:", tensor_data_z)

dataloader_z = torch.utils.data.DataLoader(tensor_data_z, shuffle= False, batch_size=1)
in1, l1 = next(iter(dataloader_z))
print(in1, l1)
# Set training parameters
model.debug = True
model.set_training_parameters(nn.MSELoss(), optimizer)

model.fit(dataloader=dataloader_z,
          num_epochs=1,
          log_interval=1)

#%%
# Check if network got to 3x+b
print("Trained weights")
#print(list(model.parameters()))
model.view_parameters()

# Test prediction from network
print("Prediction")
prediction = model(torch.ones(2, 2)) #rows, columns
print(prediction, prediction.size())
#%%

