import numpy as np
from sklearn.metrics import auc
import math
import torch

def generate_conf_matrix(gt_image, pre_image, num_class):
    mask = (gt_image >= 0) & (gt_image < num_class)  # 1 is inside the classes
    # 19 * (0-18) + pre_image
    label = num_class * gt_image[mask].astype('int') + pre_image[mask]  # mask-gt and pre-image
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def generate_conf_matrix_torch(gt_image, pre_image, num_class):
    mask = (gt_image >= 0) & (gt_image < num_class)  # 1 is inside the classes
    # 19 * (0-18) + pre_image
    label = num_class * gt_image[mask] + pre_image[mask]  # mask-gt and pre-image
    count = torch.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix.float()


class Evaluator(object):
    def __init__(self, num_class):
        """
        Evaluate the segmentation map but it must be
        in a numpy format.

        :param num_class: int
        """
        self.num_class = num_class
        self.confusion_matrix = torch.zeros((self.num_class,)*2).cuda()

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-6)
        return Acc.item()

    def Pixel_Accuracy_Class(self, each_class=False):
        Acc = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(dim=1)  + 1e-6)
        if not each_class:
            Acc = torch.mean(Acc)
        return Acc.item()

    def Mean_Intersection_over_Union(self, each_class=False):
        MIoU = torch.diag(self.confusion_matrix) / (self.confusion_matrix.sum(0) + self.confusion_matrix.sum(1) - torch.diag(self.confusion_matrix) + 1e-8)
        if not each_class:
            MIoU = torch.mean(MIoU)
            return MIoU.item()
        else:
            return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = torch.sum(self.confusion_matrix, dim=1) / torch.sum(self.confusion_matrix)
        iu = torch.diag(self.confusion_matrix) / (
                    torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                    torch.diag(self.confusion_matrix) + 1e-8)

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU.item()

    def _generate_matrix(self, gt_image, pre_image):
        return generate_conf_matrix_torch(gt_image, pre_image, self.num_class)

    def add_conf_matrix(self, matrix):
        # per frame Miou
        self.confusion_matrix += matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).cuda()

class CM_Evaluator(object):
    def __init__(self, f1_beta=1, threshold=[-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.9]):
        """
        Evaluate the segmentation map but it must be
        in a numpy format.

        :param num_class: int
        """
        self.num_class = 2
        # ERROR IS POSTIVE
        # TN FP
        # FN TP
        self.threshold = threshold
        self.f1_beta = f1_beta
        self.num_threshold = len(self.threshold)
        self.confusion_matrix = np.zeros((self.num_threshold, 2,2))

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)        # 1 is inside the classes
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask] # mask-gt and pre-image
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for i in range(self.num_threshold):
            pre_image_thresholded = pre_image > self.threshold[i]
            self.confusion_matrix[i] += self._generate_matrix(gt_image, pre_image_thresholded)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_threshold, self.num_class, self.num_class))

    def _recall(self):
        val = np.zeros((self.num_threshold, self.num_class))
        for i in range(self.num_threshold):
            val[i] = np.diag(self.confusion_matrix[i]) / (self.confusion_matrix[i].sum(axis=1) + 1e-6)
        return val.swapaxes(0,1)

    def _precision(self):
        val = np.zeros((self.num_threshold, self.num_class))
        for i in range(self.num_threshold):
            val[i] = np.diag(self.confusion_matrix[i]) / (self.confusion_matrix[i].sum(axis=0) + 1e-6)
        return val.swapaxes(0,1)

    def tpr(self, axis=1):
        return self._recall()[axis]

    def fpr(self, axis=1):
        if axis == 0: # it
            axis = 1
        else:
            axis = 0

        val = np.zeros((self.num_threshold, self.num_class))
        for i in range(self.num_threshold):
            val[i] = np.diag(np.fliplr(self.confusion_matrix[i])) / (self.confusion_matrix[i].sum(axis=1) + 1e-6)
        return val.swapaxes(0,1)[axis]

    def aupr(self, axis=1):

        x = self._precision()[axis]
        y = self._recall()[axis]
        # if axis == 0:
        #     x = np.flip(x)
        #     y = np.flip(y)
        return auc(y, x) # recall, precision



    def auc(self, axis=0):
        x = self.tpr(axis)
        y = self.fpr(axis)
        if axis == 1:
            x = np.flip(x,axis=0)
            y = np.flip(y, axis=0)
        x = np.append(x, 1)
        x = np.insert(x, 0, 0)
        y = np.append(y, 1)
        y = np.insert(y, 0, 0)
        return auc(y, x) # fpr, tpr


    def id_threshold(self):
        all_miou = []
        for i in range(self.num_threshold):
            TMIoU = np.diag(self.confusion_matrix[i]) / (
                        np.sum(self.confusion_matrix[i], axis=1) + np.sum(self.confusion_matrix[i], axis=0) -
                        np.diag(self.confusion_matrix[i]))
            TMIoU = np.nanmean(TMIoU)
            all_miou.append(TMIoU)
        return int(np.argmax(all_miou))

    def best_threshold(self):
        return self.threshold[self.id_threshold()]

    def all_f1(self, index=1):
        precision =self._precision()[index]
        recall = self._recall()[index]
        beta_sq =  math.pow(self.f1_beta,2)
        f1 = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-6)
        return f1

    def f1(self, index=1, beta=1):
        return np.max(self.all_f1(index))

    def acc(self, axis=1):
        id_threshold = self.id_threshold()
        Acc = np.diag(self.confusion_matrix[id_threshold]).sum() / self.confusion_matrix[id_threshold].sum()
        return Acc

    def Mean_Intersection_over_Union(self):
        all_miou = []
        for i in range(self.num_threshold):
            TMIoU = np.diag(self.confusion_matrix[i]) / (
                        np.sum(self.confusion_matrix[i], axis=1) + np.sum(self.confusion_matrix[i], axis=0) -
                        np.diag(self.confusion_matrix[i]))
            TMIoU = np.nanmean(TMIoU)
            all_miou.append(TMIoU)
        return max(all_miou)
