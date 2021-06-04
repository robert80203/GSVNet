import torch
import torch.nn as nn
import torch.nn.functional as F
from model.intra.networks import IntraNetwork
from model.guide.networks import GuidingNetwork


class IdealDelayKernels(nn.Module):
    def __init__(self, kernel_size, out_channel, is_train_filter=True):
        super().__init__()
        filter_kernel_s = torch.zeros((out_channel, 1 ,1 , kernel_size, kernel_size), requires_grad=False).cuda()
        #set idk for ideal-delay kenel, else random initialization
        init_mode = 'idk'

        if init_mode == 'idk':
            idx = -1
            offset = (kernel_size - 3)//2
            for i in range(offset, kernel_size - offset):
                for j in range(offset, kernel_size - offset):
                    if (i == kernel_size//2) and (j == kernel_size//2):
                        continue
                    idx += 1
                    filter_kernel_s[idx, 0, 0, i, j] = 1.0
            #point initialization
            if out_channel > 9:
                offset = (kernel_size - 3)//2 + 1
                for i in range(0,kernel_size,offset):
                    for j in range(0,kernel_size,offset):
                        if (i == kernel_size//2) and (j == kernel_size//2):
                            continue
                        idx += 1
                        filter_kernel_s[idx, 0, 0, i, j] = 1.0
        else:
            filter_kernel_s.normal_(mean=0,std=0.02)

        self.filter_copy_s = nn.Parameter(filter_kernel_s, requires_grad=is_train_filter)
        self.conv3d_padding = (kernel_size-1)//2

    def copy_conv(self,x):
        short_range = F.conv3d(x, self.filter_copy_s, padding=(0,self.conv3d_padding,self.conv3d_padding), dilation=1)
        x = torch.cat((x,short_range),dim=1)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.copy_conv(x)
        return x

class GuidedSpatiallyVaryingConv(nn.Module):
    def __init__(self, in_class, freeze_bn=True, out_class=None, num_filter=16,
                 batch_norm=nn.BatchNorm2d, is_train_filter=False):
        super().__init__()
        
        if out_class is None: out_class = in_class
        self.num_filter  = num_filter
        self.is_train_filter = is_train_filter

        self.cpsize = cpsize = 3
        self.dilation_short = 1
        #for ideal-delay kernel. e.g. 3 x 3 = 9
        self.move_filter = 9
        self.guide_filter = 32
        INTRA_in_channel = 3
        #number of classes + edge map
        GN_in_channel = 19 + 1 
        
        #batch_norm = build_bn(batch_norm, 'identity')
        guide_norm = batch_norm

        self.ideal_dk = IdealDelayKernels(cpsize, self.move_filter, self.is_train_filter)
        self.filter_lapla = torch.ones((1,1,3,3), requires_grad=False).cuda()
        self.filter_lapla[0,0,1,1] = -8
        self.intra_net = IntraNetwork(INTRA_in_channel, self.num_filter, 19, batch_norm)
        #2 = 1 + 1 (one is for intra feature, the other one is for skip connection)
        #skip connection is involved in idk, but we add an extra one (a little bit different from the paper)
        GN_out_channel = self.move_filter + 2
        self.guide = GuidingNetwork(GN_in_channel, self.guide_filter, GN_out_channel, batch_norm)
        self.init_weight_xavier()
        if freeze_bn: self.freeze_bn()

    def forward(self, x, image):
        #generate intra feature map
        intra_feature = self.intra_net(image)
        
        #generate guiding dynamic filters
        guide_input = intra_feature.squeeze(1)
        single_x = torch.argmax(x, dim=1, keepdim=True).float()
        edge = F.conv2d(single_x, self.filter_lapla, padding=1)#image boundary will be included
        guide_feat, edge = self.guide(guide_input, edge)
        guide_feat = guide_feat / (torch.sum(guide_feat, dim=1, keepdim=True) + 1e-9)
        
        #ideal-delay filtering
        x = self.ideal_dk(x)
        
        #perform convolution on intra feature map + shift maps with dynamic filters
        x = torch.cat((x, intra_feature),dim=1)
        feat = guide_feat * x
        x = torch.sum(feat, dim=1)#summation
        return x, guide_feat, edge
    
    def weights_init(self):
        for module in self.modules():
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.1, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(0.1, 0.02)
                    m.bias.data.fill_(0)

    def init_weight_xavier(self):
        for module in self.modules():
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.xavier_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, InPlaceABNSync):
                m.eval()
            elif isinstance(m, InPlaceABN):
                m.eval()
            elif isinstance(m, ABN):
                m.eval()
    