import torch.nn as nn
from .resnet import resnet50


class OctExtractor(nn.Module):

    def __init__(self):
        super(OctExtractor, self).__init__()
        r50 = resnet50(pretrained=False)
        layers = []
        for name, module in r50.named_children():
            if name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1']:
                if name == 'layer1':
                    layers.append(module[:-1])
                else:
                    layers.append(module)
        self.conv1 = nn.Sequential(*layers)

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(
                64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64,
                64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False),
            nn.BatchNorm2d(
                64,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
