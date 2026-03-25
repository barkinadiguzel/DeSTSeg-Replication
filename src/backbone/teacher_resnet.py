import torch
import torch.nn as nn
from torchvision.models import resnet18

class TeacherResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained=pretrained)
        self.blocks = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3])

    def forward(self, x):
        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x)
        return features  
