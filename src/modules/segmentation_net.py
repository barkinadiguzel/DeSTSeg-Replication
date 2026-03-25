import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, in_channels=3*256, out_channels=1):
        super().__init__()
        self.resblock1 = nn.Sequential(
            nn.Conv2d(in_channels, 256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(inplace=True)
        )
        self.aspp = nn.Conv2d(256, out_channels,1)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.aspp(x)
        return x
