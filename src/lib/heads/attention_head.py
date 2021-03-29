import torch
import torch.nn as nn
import numpy as np

class AttentionBlock(nn.Module):

    def __init__(self, in_channels=80):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)

        return x

class AttentionNet(nn.Module):

    def __init__(self):
        super(AttentionNet, self).__init__()
        self.att = AttentionBlock()

    def forward(self, x1, x2):
        a1 = self.att(x1)
        a2 = self.att(x2)
        mean_fea = (a1 * x1 + a2 * x2) / (a1 + a2)

        return mean_fea
