import torch
from model.swiftnet.models.semseg import SemsegModel
from model.swiftnet.models.resnet.resnet_single_scale import *

def Swiftnet(num_classes, batchnorm):
    mean = [73.15, 82.90, 72.3]
    std = [47.67, 48.49, 47.73]
    resnet = resnet18(pretrained=True, efficient=False, mean=mean, std=std, scale=1, batchnorm=batchnorm)
    model = SemsegModel(resnet, num_classes, batchnorm=batchnorm)
    return model