import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraNetwork(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, batchnorm):
        super().__init__()
        self.intra_net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            batchnorm(mid_ch),
            nn.ReLU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            batchnorm(mid_ch),
            nn.ReLU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1),
        )

    def forward(self, x):
        return self.intra_net(x).unsqueeze(1)