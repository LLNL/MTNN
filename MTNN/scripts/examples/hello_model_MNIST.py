# !/usr/bin/env/ python
""" mtnnpython/examples/hello_model_MNIST.py
Code to compare 1 layer  fully-cnnected layer MTNN.Model object with a simple native Torch model
- using MNIST data
- without MTNN core
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
import torch.utils as utils
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# local source
import MTNN
from scratches.codebase import constants

# Set-up logger.
logging.basicConfig(filename= constants.EXPERIMENT_LOGS_DIR + "/" + constants.get_caller_filepath() + "_"
                              + datetime.datetime.today().strftime("%A") + "_"
                              + datetime.datetime.today().strftime("%m%d%Y") + "_"
                              + datetime.datetime.now().strftime("%H:%M:%S")
                              + ".log.txt",
                    filemode='w',
                    format='%(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# Redirecting stdout to file in MTNN/examples/runs/logs
FILEOUT = open(constants.EXPERIMENT_LOGS_DIR
               + "/" + constants.get_caller_filepath() + "_"
               + datetime.datetime.today().strftime("%A") + "_"
               + datetime.datetime.today().strftime("%m%d%Y") + "_"
               + datetime.datetime.now().strftime("%H:%M:%S")
               + ".stdout.txt", "w")

sys.stdout = FILEOUT

##################################################
# Simple fully-connected network
##################################################
input_size = 784
hidden_layer_size = 1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # By default, bias is true
        self.fc1 = nn.Linear(input_size, hidden_layer_size) # row, column # input, output



    def forward(self, x):
        x = self.fc1(x)
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


def print_prediction(model, input:tuple):
    """ Prints out model weights and the prediction given the input.
    Args:
        model: <torch.Net> or <MTNN.model>
        input: <tuple>

    Returns: None

    """
    # See if network weights converged to linear fn Wx+b
    print("\nTRAINED WEIGHTS")
    model.parameters()

    # Print prediction from network
    print("\nPrediction")
    prediction =model(torch.ones(input))  # rows, columns
    print(prediction, prediction.size())


#########################################################
# Training Data
#########################################################
#sklearn linear function generate - regression training data
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


# lambda generated function
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

######################################################
# MNIST training data
#####################################################
# Load and transform data
TRANSFORM_FN = transforms.Compose(
    [transforms.Resize((28, 28)), # flatten images
     transforms.ToTensor(),  # convert image to a PyTorch tensor
     transforms.Normalize((0.1307,), (0.3081,))])  # normalize with mean (tuple), standard deviation (tuple)

# Training data
TRAIN_DATASET = datasets.MNIST(root = './datasets',
                               train = True,
                               download = True,
                               transform = TRANSFORM_FN)

print(TRAIN_DATASET)
TRAINLOADER = utils.data.DataLoader(TRAIN_DATASET,
                                    batch_size = constants.BATCH_SIZE_TRAIN,
                                    shuffle = True,
                                    num_workers = 2)  # multi-process data loading

# Testing data
TEST_DATASET = datasets.MNIST(root = './datasets',
                              train = False,
                              download = True,
                              transform = TRANSFORM_FN)
TESTLOADER = utils.data.DataLoader(TEST_DATASET,
                                   batch_size = constants.BATCH_SIZE_TEST,
                                   shuffle = False,
                                   num_workers = 2)


#########################################################
# Using Simple net class
#########################################################
print("\n\n*****************************")
print("Using Simple net Class")
print("*****************************")

# Set-up
net = Net()
print("\nNET Parameters", list(net.parameters()))

optimizer = optim.SGD(net.parameters(), lr= constants.LEARNING_RATE, momentum= constants.MOMENTUM)


# Train.
total_step = len(TRAINLOADER)
for epoch in range(constants.N_EPOCHS): #TODO: account for this in MTNN Model training
    for batch_idx, data in enumerate(TRAINLOADER, 0):
        inputs, targets = data

        print("inputs", inputs.size())
        print(inputs)

        #targets = targets.squeeze(1)
        print("labels", targets.size())
        print(targets)

        optimizer.zero_grad()
        outputs = net(inputs)  # outputs is predicted Y

        print("predicted outputs", outputs)
        print("labels", targets)
        loss = F.cross_entropy(outputs, targets)

        # Tensorboard
        visualize(net, inputs, loss, epoch)

        loss.backward() # loss.backward() equivalent to loss.backward(torch.Tensor([1]))
                         # only valid if tensor contains a single element/scalar

        # Update weights
        optimizer.step()

        # Print out
        #if batch_idx % (len(training_data) - 1) == 0 and batch_idx != 0:
        print("Epoch {} - loss: {}".format(epoch, loss.item ()))
        if (batch_idx + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, constants.N_EPOCHS, batch_idx + 1, total_step, loss.item()))

        """
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.
        """


# Predict.
print_prediction(net, (2, 2)) #should be 5

###########################################################
# Using MTNN Model
##########################################################
print("\n\n*****************************")
print("Using MTNN Model")
print("*****************************")

# Set-up.
model_config = os.path.join(constants.CONFIG_DIR + "/hello_model_mnist.yaml")
mtnnmodel = MTNN.Model(visualize =False, debug=True)
mtnnmodel.set_config(model_config)
model_optimizer = optim.SGD(mtnnmodel.parameters(), lr = constants.LEARNING_RATE, momentum = constants.MOMENTUM)
mtnnmodel.set_training_parameters(objective=nn.MSELoss(), optimizer=model_optimizer)

# View parameters.
mtnnmodel.print_parameters()

# Train.
mtnnmodel.fit(dataloader=TRAINLOADER, num_epochs=10, log_interval=10)

# Predict.
print_prediction(mtnnmodel, (2,2)) # should be 5

#########################################################
# Using Prolonged Model
#########################################################
print("\n\n*****************************")
print("USING PROLONGED MTNN MODEL")
print("*****************************")
# Applying Lower Triangular Operator
prolongation_operator = MTNN.LowerTriangleOperator()
prolonged_model = prolongation_operator.apply(mtnnmodel, exp_factor =3)
prolonged_model.set_debug(True)

# Set-up.
prolonged_model_optimizer = optim.SGD(prolonged_model.parameters(), lr = constants.LEARNING_RATE, momentum = constants.M
                                      )
prolonged_model.set_training_parameters( objective=nn.MSELoss(), optimizer=prolonged_model_optimizer)

# View Parameters.
prolonged_model.print_parameters()

# Train.
prolonged_model.fit(dataloader=TRAINLOADER, num_epochs = 10, log_interval = 10)

# Predict.
print_prediction(prolonged_model, (2,2)) # should be 5

############################################################
# Evaluate Results
#############################################################
print("\n\n*****************************")
print("EVALUATION")
print("*****************************")
print("FUNCTION 3x + 2y")
evaluator = MTNN.BasicEvaluator()
print("Net")
evaluator.evaluate_output(model=net, dataset=TEST_DATASET)
print("MTNN Model")
evaluator.evaluate_output(model=mtnnmodel, dataset=TEST_DATASET)
print("Prolonged Model")
evaluator.evaluate_output(model=prolonged_model, dataset=TEST_DATASET)

FILEOUT.close()