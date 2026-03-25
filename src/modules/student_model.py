import torch
import torch.nn as nn
from ..backbone.student_encoder import StudentEncoder
from ..layers.decoder_block import DecoderBlock

class StudentNet(nn.Module):
    def __init__(self, feature_channels=[64,128,256,512]):
        super().__init__()
        self.encoder = StudentEncoder()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(feature_channels[3], feature_channels[2]),
            DecoderBlock(feature_channels[2], feature_channels[1]),
            DecoderBlock(feature_channels[1], feature_channels[0])
        ])

    def forward(self, x):
        enc_feats = self.encoder(x)
        x = enc_feats[-1]  
        dec_feats = []
        for i, dec in enumerate(self.decoder_blocks):
            x = dec(x)
            dec_feats.append(x)
        return dec_feats  
