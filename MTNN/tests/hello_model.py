# !/usr/bin/env/ python
"""
Sample code to test MTNN.Model class with a simple native Torch model
Note: Best to run this in a Jupyter notebook environment to observe each step.
"""
# system packages
import datetime
import yaml

# pytorch packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary

# local packages
import MTNN
from MTNN import LowerTriangleOperator

##################################################
# Simple fully-connected network
##################################################
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


def visualize(model, input, loss, epoch):
    # Clear previous runs

    writer = SummaryWriter('./runs/net/' + str(datetime.datetime.now()))
    # Record training loss from each epoch into the writer
    writer.add_scalar('Train/Loss', loss.item(), epoch)

    writer.flush()
   # Write to computation graph
    writer.add_graph(model, input)
    writer.flush()

# Creating dummy data for training
z = lambda x,y: 3 * x + 2 *y
data_z = gen_data()
print(data_z)

"""
# Using Simple net class
net = Net()
print(net)
print(list(net.parameters()))

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)


# Run the main training loop
for epoch in range(10):
    for i, data in enumerate(data_z):
        # unpack
        XY, Z = iter(data)
        # print(X, Y, Z)
        # Tensorize input and wrap in Variable to apply gradient descent
        # requires_grad is False by default

        # Convert list to float tensor
        # Variable is deprecated
        input = torch.FloatTensor(XY) #torch.Floattensor expects a list

        Z = torch.FloatTensor([Z])   # For output, set requires_grad to False
        # print("Input", input, input.size(),"output", Z)
        # Zero out previous gradients
        optimizer.zero_grad()
        outputs = net(input)  # outputs is predicted Y
        # print("OUTPUT", outputs, outputs.size())
        #print("WEIGHTS", net.fc1.weight)

        loss = F.mse_loss(outputs, Z)
        #print("LOSS", loss)

        visualize(net,input,loss,epoch)

        loss.backward() # loss.backward() equivalent to loss.backward(torch.Tensor([1]))
                         # only valid if tensor contains a single element/scalar
        #print("GRADIENTS", net.fc1.weight.grad)

        # Update weights
        optimizer.step()
        if i % (len(data_z) - 1) == 0 and i != 0:  # Check me please
            print("Epoch {} - loss: {}".format(epoch, loss.item ()))


# Check if network got to 3x+ 3y +b
print("Trained weights")
print(list(net.parameters()))

# Test prediction from network
print("Prediction")
prediction = net(torch.ones(2, 2)) #rows, columns
print(prediction, prediction.size())

"""

# Using the MTNN model
# Using test.yaml config file
model_config = yaml.load(open("/Users/mao6/proj/mtnnpython/MTNN/tests/test.yaml", "r"), Loader=yaml.SafeLoader)
model = MTNN.Model(tensorboard=True, debug=False)
model.set_config(model_config)

print(model.parameters())
model_optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
model.set_training_parameters( objective=nn.MSELoss(), optimizer=model_optimizer)


print(model)
print(model.view_parameters())

###########################################################
# Testing on MTNN Model
##########################################################
print("Using MTNN Model")

# Load Data_z into Pytorch dataloader
tensor_data_z = []
for i, data in enumerate(data_z):
    XY, Z = iter(data)
    # Convert list to float tensor
    input = Variable(torch.FloatTensor(XY), requires_grad = False) # Note: torch.Floattensor expects a list
    Z = torch.FloatTensor([Z])
    tensor_data_z.append((input, Z))

print("Data:", tensor_data_z)
dataloader_z = torch.utils.data.DataLoader(tensor_data_z, shuffle= False, batch_size=1)

# Train.
model.fit(dataloader=dataloader_z, num_epochs=10,log_interval=10)


# Check if network weights converged to lambda fn 3x+b
print("Trained weights")
model.view_parameters()  # should be 3,2

# Test prediction from network
print("Prediction")
prediction = model(torch.ones(2, 2))  # rows, columns
print(prediction, prediction.size())  # should be 5


# Testing Lower Triangular Operator
prolongation_operator = MTNN.LowerTriangleOperator()
prolonged_model = prolongation_operator.apply(model, expansion_factor=3)
prolonged_model.__setattr__('debug', False) #TODO: Make a mutator method to set debug (insetad of using magic method)

prolonged_model_optimizer = optim.SGD(prolonged_model.parameters(), lr = 0.01, momentum = 0.5)
prolonged_model.set_training_parameters( objective=nn.MSELoss(), optimizer=prolonged_model_optimizer)
prolonged_model.fit(dataloader = dataloader_z, num_epochs = 1, log_interval = 10)


# Train on prolonged model.
prolonged_model.__setattr__('debug', False)
prolonged_model.fit(dataloader=dataloader_z, num_epochs=10,log_interval=10)