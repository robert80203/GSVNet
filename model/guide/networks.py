import torch
import torch.nn as nn


class GuidingNetwork(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, batchnorm):
        super().__init__()
        self.guide_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            batchnorm(mid_ch),
            nn.ReLU(),
        )  
        self.guide_conv_feat = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch*2, kernel_size=3, stride=2, padding=1, dilation=1),
            batchnorm(mid_ch*2),
            nn.ReLU(),
            nn.Conv2d(mid_ch*2, mid_ch*4, kernel_size=3, stride=2, padding=1, dilation=1),
            batchnorm(mid_ch*4),
            nn.ReLU(),
        )
        self.guide_deconv = nn.Sequential(
            nn.Upsample(scale_factor=2.0,mode='bilinear',align_corners=True),
            nn.Conv2d(mid_ch*4, mid_ch*2, kernel_size=3, stride=1, padding=1, dilation=1),
            batchnorm(mid_ch*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.0,mode='bilinear',align_corners=True),
            nn.Conv2d(mid_ch*2, mid_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            batchnorm(mid_ch),
            nn.ReLU(),
        )
        self.guide_decoder = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            batchnorm(out_ch),
            nn.ReLU(),
        )
            
    def forward(self, x, edge):
        edge[edge!=0.0] = 1.0
        x = torch.cat((x, edge), dim=1)
        guide_feat = self.guide_conv(x)
        inter_feat = self.guide_conv_feat(guide_feat)
        inter_feat = self.guide_deconv(inter_feat)
        guide_feat = guide_feat + inter_feat
        guide_feat = self.guide_decoder(guide_feat)
        return guide_feat.unsqueeze(2), edge