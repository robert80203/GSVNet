import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
#from ptflops import get_model_complexity_info

"""
    Modified version of FlowNet 1
    Source: https://github.com/ClementPinard/FlowNetPytorch
"""
__all__ = [
    'flownets', 'flownets_bn', 'flow2rgb'
]

"""
    Visualize the Flow
    Original File: https://github.com/ClementPinard/FlowNetPytorch/blob/master/main.py
"""

import numpy as np


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


"""
    util.py
    Original Network: https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/util.py
"""


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )

    else:

        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)  # mxnet no_bias = False line 1774


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=0, bias=True),
        # padding 1 to 0 bias false to true
        # nn.LeakyReLU(0.1,inplace=True)  in mxnet crop fisrt , then do relu
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


"""
    The Network
    Original File: https://github.com/ClementPinard/FlowNetPytorch/blob/master/models/FlowNetS.py
"""


class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True, ft=False):#old version
        super(FlowNetS, self).__init__()

        self.ft = ft
        print(batchNorm, ft)
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, 6, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)  # mxnet line 1778
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)  # mxnet line 1785
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)  # mxnet line 1793
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 0, bias=True)  # mxnet line 1800
        self.avgpool = nn.AvgPool2d(2, 2)
        # self.scale_conv = nn.Conv2d(194, 1024, 1, 1, 0, bias=True)

        self.relu = nn.LeakyReLU(0.1,inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):

        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6 = self.predict_flow6(out_conv6)
        flow6_up = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)
        out_deconv5 = self.relu(out_deconv5)  # mxnet line 1777

        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)
        out_deconv4 = self.relu(out_deconv4)  # mxnet line 1784

        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)
        out_deconv3 = self.relu(out_deconv3)  # mxnet line 1791

        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)
        out_deconv2 = self.relu(out_deconv2)  # mxnet line 1798

        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        # miss a avgpooling layer in mxnet line 1802
        concat2 = self.avgpool(concat2)
        flow2 = self.predict_flow2(concat2)
        # miss scale  and sclae_bias layer in mxnet line 1805 1805
        # flow2_scale = self.scale_conv(concat2)
        return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

class FlowLight(nn.Module):
    def __init__(self, batchnorm):#old version
        super(FlowLight, self).__init__()
        channels = 32
        self.conv1 = nn.Sequential(
            #nn.Conv2d(19+3,channels,kernel_size=3,stride=2,padding=1),
            nn.Conv2d(3+3,channels,kernel_size=3,stride=2,padding=1),
            batchnorm(channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels,channels*2,kernel_size=3,stride=2,padding=1),
            batchnorm(channels*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_dilate_1 = nn.Sequential(
            nn.Conv2d(channels*2,channels*2,kernel_size=3,stride=1,padding=1),
            batchnorm(channels*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_dilate_2 = nn.Sequential(
            nn.Conv2d(channels*2,channels*2,kernel_size=3,stride=1,padding=2,dilation=2),
            batchnorm(channels*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_dilate_3 = nn.Sequential(
            nn.Conv2d(channels*2,channels*2,kernel_size=3,stride=1,padding=4,dilation=4),
            batchnorm(channels*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_dilate_4 = nn.Sequential(
            nn.Conv2d(channels*2,channels*2,kernel_size=3,stride=1,padding=8,dilation=8),
            batchnorm(channels*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_merge = nn.Sequential(
            nn.Conv2d(channels*4*2,channels*2,kernel_size=1,stride=1,padding=0),
            batchnorm(channels*2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.toflow = nn.Conv2d(channels*2, 2, kernel_size=3, stride=1, padding=1)
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x_1 = self.conv_dilate_1(x)
        x_2 = self.conv_dilate_2(x) + x_1
        x_3 = self.conv_dilate_3(x) + x_2
        x_4 = self.conv_dilate_4(x) + x_3
        out = self.conv_merge(torch.cat((x_1,x_2,x_3,x_4),dim=1))
        out = out + x
        out = self.toflow(out)
        return out

def Flownets(name, path, batchnorm=nn.BatchNorm2d):
    if name == 'light':
        print('Load our lightweight network...')
        model = FlowLight(batchnorm)
    elif name == 'flownet':
        print('Load flownets with pretrained weights...')
        model = FlowNetS(batchNorm=False)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt,  False)
    else:
        raise NotImplementedError("Optical flow network checkpoint is not recognized.")

    return model
