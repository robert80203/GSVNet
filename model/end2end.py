import torch
import torch.nn.functional as F
import torch.nn as nn


class End2End(nn.Module):
    def __init__(self, model):
        super(End2End, self).__init__()
        self.grm = model
    def forward(self, mc_output, resi_or_image):
        pred = self.grm(mc_output, resi_or_image)
        return pred