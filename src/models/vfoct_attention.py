import torch
import torch.nn as nn
from extractor.oct_extractor import OctExtractor
from extractor.vf_extractor import VfExtractor
from heads.mlp_head import FCNet
from heads.attention_head import AttentionNet

class VfOctAttentionNet(nn.Module):

    def __init__(self):
        super(VfOctAttentionNet, self).__init__()
        self.octnet = OctExtractor()
        self.vfnet = VfExtractor()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = AttentionNet()
        self.fcnet = FCNet(in_channels=80)

    def forward(self, oct_in, vf_in):
        oct_fea = self.octnet(oct_in)
        oct_fea = self.avgpool(oct_fea)

        vf_fea = self.vfnet(vf_in)
        vf_fea = self.avgpool(vf_fea)

        fusion_fea = self.alpha(oct_fea, vf_fea)
        out = self.fcnet(fusion_fea)
        
        return out
