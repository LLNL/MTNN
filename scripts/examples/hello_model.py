# !/usr/bin/env/ python
""" scripts/examples/hello_model.py
Code to compare 1 fully-cnnected layer MTNN.Model object with a simple native Torch model
- using generated linear regression data 3x + 2y
- without MTNN framework
"""
# standard
import os
import datetime
import logging
import sys

# third-party
import sklearn.datasets as skdata

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# local source
import MTNN
from MTNN import mtnn_defaults
import MTNN.logger as logger


# Set-up logger.
logging.basicConfig(filename=(mtnn_defaults.EXPERIMENT_LOGS_FILENAME + ".log.txt"),
                    filemode='w',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# Redirecting stdout to file in MTNN/examples/runs/logs
"""
FILEOUT = open(mtnn_defaults.EXPERIMENT_LOGS_DIR
               + "/" + mtnn_defaults.get_caller_filename() + "_"
               + datetime.datetime.today().strftime("%A") + "_"
               + datetime.datetime.today().strftime("%m%d%Y") + "_"
               + datetime.datetime.now().strftime("%H:%M:%S")
               + ".stdout.txt", "w")

#sys.stdout = FILEOUT
"""
sys.stdout = logger.StreamLogger()

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

####################################################
# Helper functions.
######################################################

def visualize(model, input, loss, epoch):
    """ Records control and data flow graph to Pytorch TensorboardX.
    See MTNN/README.md on how to use TensorboardX
    Args:
        model:
        input:
        loss:
        epoch:

    Returns: None

    """
    # Clear previous runs
    writer = SummaryWriter('./runs/net/' + str(datetime.datetime.now()))

    # Record training loss from each epoch into the writer
    writer.add_scalar('Train/Loss', loss.item(), epoch)
    writer.flush()

   # Write to computation graph
    writer.add_graph(model, input)
    writer.flush()


def print_prediction(model, input_value: tuple):
    """ Prints out model weights and the prediction given the input.
    Args:
        model: <torch.Net> or <MTNN.model>
        input_value: <tuple>

    Returns: None

    """
    # Summary of parameters


    # See if network weights converged to linear fn Wx+b


    try: # For MTNN Model
        print("\nTRAINED WEIGHTS")
        model.print_parameters()
    except:
        summary(model, input_value)
        print("\nTRAINED WEIGHTS")
        for param in model.parameters():
            print(param.data)

    # Input
    print(f'\nINPUT:{input_value}')

    # Print prediction from network
    print("\nPrediction")
    prediction = model(torch.ones(input_value))  # rows, columns
    print(prediction, prediction.size())



#########################################################
# Preparing Training Data
#########################################################
def regression_training_data(num_samples: int, num_features: int, noise_level:float) -> list:
    """
    Returns a list of tuples a tuple of tensor training input and output data
    from a randomly generated regression problem.
    Args:
        num_samples: <int>
        num_features: <int>
        noise_level: <float>

    Returns:
        training_data_input <tensor>
        training_data_output <tensor>
    """
    regression_data = []
    print("\nSETUP: Generating regression training data")
    x, y = skdata.make_regression(n_samples=num_samples,
                                  n_features=num_features,
                                  noise=noise_level)
    # Tensorize data.
    for i in range(len(x)):
        training_data_input = Variable(torch.FloatTensor(x), requires_grad = True)
        training_data_output = Variable(torch.FloatTensor(x))
        regression_data.append((training_data_input, training_data_output))

    #print(regression_data)
    return regression_data

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
        input = Variable(torch.FloatTensor(XY), requires_grad = False)  # Note: torch.Floattensor expects a list
        Z = torch.FloatTensor([Z])

        tensor_data.append((input, Z))

    dataloader = torch.utils.data.DataLoader(tensor_data, shuffle = False, batch_size = 1)

    return dataloader


def gen_data(linear_function):
    generated_data = []
    for i in range(10):
        input_data = ((i, i), linear_function(i, i))
        generated_data.append(input_data)
    return generated_data

# Generate Data.
linear_function = lambda x, y: 3 * x + 2 * y
training_data = gen_data(linear_function)

tensor_training_data = tensorize_data(training_data)
#########################################################
# Using Simple net class
#########################################################
print("\n\n*****************************")
print("Using Simple net Class")
print("*****************************")

net = Net()
print("\nNET Parameters", list(net.parameters()))

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# TODO: Clean
# Train.
for epoch in range(10):
    for batch_idx, data in enumerate(training_data):
        XY, Z = iter(data)

        # Convert list to float tensor
        # Note: Variable is deprecated
        input = torch.FloatTensor(XY) #torch.Floattensor expects a list
        Z = torch.FloatTensor([Z]) # For output, set requires_grad to False


        #TODO: FIX Z ValueError: only one element tensors can be converted to python scalars
        optimizer.zero_grad()
        outputs = net(input)  # outputs is predicted Y
        loss = F.mse_loss(outputs, Z)

        visualize(net, input, loss, epoch)

        loss.backward() # loss.backward() equivalent to loss.backward(torch.Tensor([1]))
                         # only valid if tensor contains a single element/scalar

        # Update weights
        optimizer.step()

        # Print out
        #if batch_idx % (len(training_data) - 1) == 0 and batch_idx != 0:
        print("Epoch {} - loss: {}".format(epoch, loss.item ()))


# Predict.
#print_prediction(net, (2, 2)) #should be 5
print("NET MODEL PARAMETERS")
for param in net.parameters():
    print(param.data)

###########################################################
# Using MTNN Model
##########################################################
print("\n\n*****************************")
print("Using MTNN Model")
print("*****************************")

# Set-up.
model_config = os.path.join(mtnn_defaults.CONFIG_DIR + "/hello_model.yaml")
mtnnmodel = MTNN.Model(tensorboard=False, debug=True)
mtnnmodel.set_config(model_config)
model_optimizer = optim.SGD(mtnnmodel.parameters(), lr = 0.01, momentum = 0.5)
mtnnmodel.set_training_parameters(objective=nn.MSELoss(), optimizer=model_optimizer)

# Train.
mtnnmodel.fit(dataloader=tensor_training_data, num_epochs=10, log_interval=10)

# View parameters.
mtnnmodel.print_parameters()

# Predict.
#print_prediction(mtnnmodel, (2,2)) # should be 5

#########################################################
# Using Prolonged Model
#########################################################
print("\n\n*****************************")
print("USING PROLONGED MTNN MODEL")
print("*****************************")
# Applying Lower Triangular Operator
prolongation_operator = MTNN.LowerTriangleOperator()
prolonged_model = prolongation_operator.apply(mtnnmodel, expansion_factor=3)
prolonged_model.set_debug(True)

# Set-up.
prolonged_model_optimizer = optim.SGD(prolonged_model.parameters(), lr = 0.01, momentum = 0.5)
prolonged_model.set_training_parameters(objective=nn.MSELoss(), optimizer=prolonged_model_optimizer)

# Train.
prolonged_model.fit(dataloader=tensor_training_data, num_epochs = 10, log_interval = 10)

# View Parameters.
prolonged_model.print_parameters()

# Predict.
#print_prediction(prolonged_model, (2, 2)) # should be 5

############################################################
# Evaluate Results
#############################################################
print("\n\n*****************************")
print("EVALUATION")
print("*****************************")
print("FUNCTION 3x + 2y")
print("NET MODEL PARAMETERS")
for param in net.parameters():
    print(param.data)
evaluator = MTNN.BasicEvaluator()
print("\nNet")
evaluator.evaluate_output(model=net, dataset=tensor_training_data)
print("\nMTNN Model")
mtnnmodel.print_parameters()
evaluator.evaluate_output(model=mtnnmodel, dataset=tensor_training_data)
print("\nProlonged Model")
prolonged_model.print_parameters()
evaluator.evaluate_output(model=prolonged_model, dataset=tensor_training_data)

