# !/usr/bin/env/ python3
""" scripts/examples/hello_model.py
Code to compare 1 fully-cnnected layer MTNN.Model object with a simple native Torch model
- using generated linear regression data 3x + 2y
- without MTNN core
"""
# standard
import datetime
import sys
import copy

# third-party
import sklearn.datasets as skdata

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# local source
import MTNN
from core import build
from logging import logger
from scratches.codebase import trainer
from cli import env_var
from configuration import reader

#############################################
# Set-up logging stdout to file
#############################################
sys.stdout = logger.StreamLogger()

##############################################
# Read from YAML Configuration File
##############################################

conf = reader.YamlConfig(env_var.CONFIG_PATH)
BATCH_SIZE_TRAIN = conf.get_property('batch_size_train')
BATCH_SIZE_TEST = conf.get_property('batch_size_test')


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
def get_regression_training_data(num_samples: int, num_features: int, noise_level: float) -> list:
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
    return regression_data


def tensorize_data(training_data: list):
    """Tensorizes data and load into a Pytorch DataLoader.
    Args:
        training_data: <list> of tuples (input, output)

    Returns:
        datasets: <torch.utils.data.DataLoader>

    """
    # Load Data_z into Pytorch datasets
    tensor_data = []
    for i, data in enumerate(training_data):
        XY, Z = iter(data)

        # Convert list to float tensor
        input = torch.tensor(XY, dtype=torch.float,  requires_grad=True)
        Z = torch.FloatTensor([Z])

        tensor_data.append((input, Z))
    dataloader = torch.utils.data.DataLoader(tensor_data, shuffle = False, batch_size = 1)

    return dataloader


def gen_data(linear_function):
    training_data = []
    input_data = []
    for x in range(3):
        for y in range(3):
            training_datum = ((x, y), linear_function(x, y))
            input_data.append((x, y))
            training_data.append(training_datum)
    return input_data, training_data

# Generate Data.
linear_function = lambda x, y: 3 * x + 2 * y
input_data, training_data = gen_data(linear_function)


#TODO: Train on SKlearn regression data
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

# Train.
for epoch in range(10):
    for batch_idx, data in enumerate(training_data):
        XY, Z = iter(data)

        # Convert list to float tensor
        input = torch.FloatTensor(XY)
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
print("NET MODEL PARAMETERS")
for param in net.parameters():
    print(param.data)

###########################################################
# Using MTNN Model with Builder
##########################################################
print("\n\n*****************************")
print("Using MTNN Model")
print("*****************************")

print("CONFIG", env_var.CONFIG_PATH)
mtnnmodel = build.build_model(env_var.CONFIG_PATH, visualize=False, debug=True)

# Build Optimizer.
optimizer = trainer.build_optimizer(env_var.CONFIG_PATH, mtnnmodel.parameters())

# Set Optimizer.
mtnnmodel.set_optimizer(optimizer)
mtnnmodel.print_properties()

print("\n MTNN MODEL PARAMETERS BEFORE TRAINING")
mtnnmodel.print_parameters()

# Train.
mtnnmodel.fit(dataloader=tensor_training_data, num_epochs=10, log_interval=10)

print("\nTRAINED MTNN MODEL PARAMETERS")
mtnnmodel.print_parameters()


#########################################################
# Using Prolonged Model
#########################################################
print("\n\n*****************************")
print("USING PROLONGED MTNN MODEL")
print("*****************************")
# Applying Lower Triangular Operator
prolongation_operator = MTNN.LowerTriangleOperator()
prolonged_model = prolongation_operator.apply(mtnnmodel, exp_factor =3)


print("\nUNTRAINED PROLONGED MTNN MODEL PARAMETERS: \n")
prolonged_model_copy = copy.deepcopy(prolonged_model)
prolonged_model.print_properties()
prolonged_model.print_parameters()

# Set-up.
prolonged_model_optimizer = optim.SGD(prolonged_model.parameters(), lr = 0.01, momentum = 0.5)
prolonged_model.set_training_parameters(objective=nn.MSELoss(), optimizer=prolonged_model_optimizer)

# Train.
prolonged_model.fit(dataloader=tensor_training_data, num_epochs = 10, log_interval = 10)


# View Parameters.
print("\nTRAINED PROLONGED MTNN MODEL PARAMETERS: \n")
prolonged_model.print_properties()
prolonged_model.print_parameters()


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

print("\nMTNN MODEL")
mtnnmodel.print_parameters()
evaluator.evaluate_output(model=mtnnmodel, dataset=tensor_training_data)

print("\nUNTRAINED PROLONGED MODEL")
prolonged_model_copy.print_parameters()
evaluator.evaluate_output(model=prolonged_model_copy, dataset=tensor_training_data)

print("\nTRAINED PROLONGED MODEL")
prolonged_model.print_parameters()
evaluator.evaluate_output(model=prolonged_model, dataset=tensor_training_data)
