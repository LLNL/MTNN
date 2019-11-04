# A class for training using SGD

import torch.optim as optim

# TODO: Maybe add "relative improvement" or something??
class SGDStoppingCriteria():
    """Stopping criteria for SGD
    
    Attributes:
    num_epochs (int): The number of passes over a finite dataset to run.
    """
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs;

class SGDTrainingAlg():
    """SGD training
    
    Attributes:
    lr_ (float): learning rate
    momentum_ (float): mass times velocity
    doprint_ (bool): whether to print "heartbeat" epoch updates
    """

    def __init__(self, lr, momentum, doprint=False):
        self.lr_ = lr
        self.momentum_ = momentum
        self.doprint_ = doprint

    def train(self, net, dataset, obj_func, criteria):
        optimizer = optim.SGD(net.parameters(), lr=self.lr_, momentum=self.momentum_)
        
        for epoch in range(criteria.num_epochs):
            running_loss = 0.0
            
            for i, data in enumerate(dataset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = net(inputs)
                loss = obj_func(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # print statistics
                if self.doprint_:
                    running_loss += loss.item()
                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
        return net
