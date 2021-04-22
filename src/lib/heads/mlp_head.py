import numpy as np
import torch.nn as nn


class FCNet(nn.Module):
    "Network head of classifier."
    
    def __init__(self, in_channels=80):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_channels, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(200, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x
