import torch
import torch.nn as nn
from torchvision.models import resnet18

class StudentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=False)
        self.blocks = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])

    def forward(self, x):
        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x)
        return features  
