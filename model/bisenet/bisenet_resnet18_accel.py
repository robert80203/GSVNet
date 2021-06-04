# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
from model.bisenet.resnet import resnet18
from model.bisenet.resnet import resnet101
from model.bisenet.opt import ConvBnRelu, AttentionRefinement, FeatureFusion


class BiSeNet(nn.Module):
    def __init__(self, out_planes, is_training,
                 criterion, ohem_criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d,in_planes = 3, real_time = 0):
        super(BiSeNet, self).__init__()
        self.real_time = real_time
        self.context_path = resnet18(pretrained_model, norm_layer=norm_layer,
                                     deep_stem=False, stem_width=64, in_planes = in_planes)

        self.business_layer = []
        self.is_training = is_training

        self.spatial_path = SpatialPath(in_planes, 128, norm_layer)

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )

        # stage = [512, 256, 128, 64]
        arms = [AttentionRefinement(512, conv_channel, norm_layer),
                AttentionRefinement(256, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]
        if self.real_time:
            if is_training:
                heads = [BiSeNetHead(conv_channel, out_planes, 2,
                                    True, norm_layer, real_time = real_time),
                        BiSeNetHead(conv_channel, out_planes, 1,
                                    True, norm_layer, real_time = real_time),
                        BiSeNetHead(conv_channel * 2, out_planes, 1,
                                    False, norm_layer, real_time = real_time)]
            else:
                heads = [None, None,
                        BiSeNetHead(conv_channel * 2, out_planes, 1,
                                    False, norm_layer, real_time = real_time)]
        else:
            heads = [BiSeNetHead(conv_channel, out_planes, 16,
                                True, norm_layer, real_time = real_time),
                    BiSeNetHead(conv_channel, out_planes, 8,
                                True, norm_layer, real_time = real_time),
                    BiSeNetHead(conv_channel * 2, out_planes, 1,
                    False, norm_layer, real_time = real_time)]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 1, norm_layer)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)

        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        if is_training:
            self.criterion = criterion
            self.ohem_criterion = ohem_criterion

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)
        context_blocks = self.context_path(data)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)
        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)


        if self.is_training and label is not None:
            #tmp = label
            #tmp = tmp.detach().cpu().numpy()
            #tmp[np.where(tmp  == 250)] = -1
            #print(tmp.max())
            label = label.cpu().numpy()
            label[np.where(label > 20)] = 250
            label = torch.Tensor(label).long().cuda()
            pred_o1 = F.interpolate(self.heads[0](pred_out[0]), size = (label.shape[1], label.shape[2]), mode='bilinear',align_corners=True)
            pred_o2 = F.interpolate(self.heads[1](pred_out[1]), size = (label.shape[1], label.shape[2]), mode='bilinear',align_corners=True)
            pred_o3 = F.interpolate(self.heads[-1](pred_out[2]), size = (label.shape[1], label.shape[2]), mode='bilinear',align_corners=True)
            aux_loss0 = self.ohem_criterion(pred_o1, label)
            aux_loss1 = self.ohem_criterion(pred_o2, label)
            main_loss = self.ohem_criterion(pred_o3, label)

            '''aux_loss0 = self.ohem_criterion(self.heads[0](pred_out[0]), label)
            aux_loss1 = self.ohem_criterion(self.heads[1](pred_out[1]), label)
            main_loss = self.ohem_criterion(self.heads[-1](pred_out[2]), label)'''

            loss = main_loss + aux_loss0 + aux_loss1
            #print(loss.item())
            return loss
        # test for end2end training
        #return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)
        return self.heads[-1](pred_out[-1]), pred_out[-1]


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d, real_time = 0):
        super(BiSeNetHead, self).__init__()
        self.real_time = real_time
        if self.real_time:
            feature_num = 128
        else:
            feature_num = 256
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, feature_num, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(feature_num, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output


if __name__ == "__main__":
    model = BiSeNet(19, is_training=False, criterion=None, ohem_criterion=None)
    print(model)
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, input_res=(3, 1024, 2048),
                                                  as_strings=True, print_per_layer_stat=True)
        print('Flops:  ' + flops)
        print('Params: ' + params)
