import torch.nn as nn


class VfExtractor(nn.Module):

    def __init__(self, in_channels=1):
        super(VfExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 120, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(120, 120, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(120, 80, kernel_size=(3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(80, 80, kernel_size=(3, 3), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.gap(x)

        return x
