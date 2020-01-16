# !/usr/bin/env/ python
""" mtnnpython/MTNN/hello_model.py
Code to test MTNN.Model class with a simple native Torch model
with generated linear regression data
Best to run this in a Jupyter notebook environment to observe each step.
"""
# standard
import datetime

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary

# local source
import MTNN


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
        #x = self.fc2(x)
        x = F.relu(x)
        return x

# Helper functions
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
data_z = gen_data()
print(data_z)


#########################################################
# Using Simple net class
#########################################################
net = Net()
print(net)
print(list(net.parameters()))

# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)


# Run the main training loop
for epoch in range(10):
    for i, data in enumerate(data_z):
        XY, Z = iter(data)

        # Convert list to float tensor
        # Note: Variable is deprecated
        input = torch.FloatTensor(XY) #torch.Floattensor expects a list

        Z = torch.FloatTensor([Z]) # For output, set requires_grad to False
        optimizer.zero_grad()
        outputs = net(input)  # outputs is predicted Y
        loss = F.mse_loss(outputs, Z)

        visualize(net, input, loss, epoch)

        loss.backward() # loss.backward() equivalent to loss.backward(torch.Tensor([1]))
                         # only valid if tensor contains a single element/scalar

        # Update weights
        optimizer.step()

        # Print out
        if i % (len(data_z) - 1) == 0 and i != 0:
            print("Epoch {} - loss: {}".format(epoch, loss.item ()))


# Check if network got to 3x+ 3y +b
print("Trained weights")
print(list(net.parameters())) # Weights should be [ 3 3]

# Test prediction from network
print("Prediction")
prediction = net(torch.ones(2, 2)) #rows, columns
print("Simple Net", prediction, prediction.size())



###########################################################
# Using MTNN Model
##########################################################
print("Using MTNN Model")

# Using the MTNN model
# Using hello_model.yaml config file
model_config ="/Users/mao6/proj/mtnnpython/MTNN/tests/hello_model.yaml"
mtnnmodel = MTNN.Model(tensorboard=True, debug=False)
mtnnmodel.set_config(model_config)

print(mtnnmodel.parameters())
model_optimizer = optim.SGD(mtnnmodel.parameters(), lr = 0.01, momentum = 0.5)
mtnnmodel.set_training_parameters(objective=nn.MSELoss(), optimizer=model_optimizer)


print(mtnnmodel)
print(mtnnmodel.view_parameters())


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

for element in dataloader_z:
    print("Input", element[0].size())
    print("Output", element[1].size())

# Train.
mtnnmodel.fit(dataloader=dataloader_z, num_epochs=10, log_interval=10)


# Check if network weights converged to lambda fn 3x+b
print("Trained weights")
mtnnmodel.view_parameters()  # should be 3,2


# Test prediction from network
print("Prediction")
prediction = mtnnmodel(torch.ones(2, 2))  # rows, columns
print(prediction, prediction.size())  # should be 5


# Testing Lower Triangular Operator
prolongation_operator = MTNN.LowerTriangleOperator()
prolonged_model = prolongation_operator.apply(mtnnmodel, expansion_factor=3)
prolonged_model.set_debug(True)

prolonged_model_optimizer = optim.SGD(prolonged_model.parameters(), lr = 0.01, momentum = 0.5)
prolonged_model.set_training_parameters( objective=nn.MSELoss(), optimizer=prolonged_model_optimizer)
prolonged_model.fit(dataloader = dataloader_z, num_epochs = 1, log_interval = 10)


# Train on prolonged model.
prolonged_model.set_debug(True)
prolonged_model.fit(dataloader=dataloader_z, num_epochs=10,log_interval=10)
prediction = mtnnmodel(torch.ones(2,2))
print("Prolonged Prediction", prediction)
############################################################
# Evaluate Results
#############################################################
print("Evaluation")
evaluator = MTNN.BasicEvaluator()

original_correct, original_total = evaluator.evaluate_output(model=net, dataset=dataloader_z)
prolong_correct, prolong_total = evaluator.evaluate_output(model=mtnnmodel, dataset=dataloader_z)


