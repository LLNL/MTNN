"""
Holds Models
"""
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def forward(self):
        raise NotImplementedError



class BasicMnistModel(BaseModel):
    """A basic image classifier."""

    def __init__(self):
        super(BasicMnistModel, self).__init__()
        self.conv1_outchan = 6
        self.conv2_outchan = 8
        self.conv1 = nn.Conv2d(3, self.conv1_outchan, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.conv1_outchan, self.conv2_outchan, 5)
        self.fc1 = nn.Linear(self.conv2_outchan * 5 * 5, 30)
        self.fc2 = nn.Linear(30, 21)
        self.fc3 = nn.Linear(21, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.conv2_outchan * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class BasicCifarModel(BaseModel):

    def __init__(self):
        super(BasicCifarModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


