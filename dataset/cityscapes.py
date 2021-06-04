import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision.transforms as standard_transforms
#from config import get_config
from torchvision.datasets.folder import default_loader

#CityscapesConfig = get_config('cityscapes_2k')

class cityscape_dataset(Dataset):
    def __init__(self, config, mode = 'train', interval = 0, label_transform = standard_transforms.ToTensor(), img_transform = standard_transforms.ToTensor(), bi_direction = False):
        self.mode = mode
        if interval < 0 :
            self.reverse = True
            self.interval = -interval
            interval = -interval
        else:
            self.reverse = False
            self.interval = interval
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.bi_direction = bi_direction
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [ 7,  8,  11, 12,13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33] # we have 19 classes + 1 void class
        self.ignore_index = config.ignore_index
        self.data_path = config.data_path
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        if mode == 'train' :
            file_path = "./dataset/cityscapes/train.txt"
        elif mode == 'val':
            file_path = "./dataset/cityscapes/val.txt"
        elif mode == 'test':
            file_path = "./dataset/cityscapes/test.txt"

        self.file_list = []
        

        with open(file_path, "r") as f :
            for line in f:
                string = line.split(" ")[0]
                label_path = line.split(" ")[1][:-1].split("/")[3]
                file_name = string[:-16]
                back_address = string[-16:]
                frame_idx = file_name[-6:]
                file_name = self.data_path+file_name[:-6]
                element = (file_name, int(frame_idx)-interval, back_address, label_path)
                self.file_list.append(element)
        
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name, start_frame, back_address, label_path = self.file_list[idx]

        imgs = []
        idxes = []
        for i in range(self.interval+1):
            if self.reverse:
                frame_num = (start_frame-i+2*self.interval) ###backward
            else:
                frame_num = (start_frame+i) ### forward
            
            img = default_loader(   os.path.join
                                (file_name + "{:06d}".format(frame_num) + back_address))   
            img = self.img_transform(img)
            imgs.append((img))
            idxes.append(frame_num)
        
        if (self.mode != 'test') and (self.mode != 'video'):
            label = default_loader(os.path.join
                                (self.data_path+"gtFine", self.mode, label_path.split("_")[0], label_path)
                                )
            #label = self.label_transform(label)
            #print(label.shape)
            label = self.encode_segmap(torch.tensor(np.array(label)[:,:,0]))
            if self.bi_direction:
                return imgs, label, idxes
            return imgs, label
        else:
            return imgs, label_path
    def encode_segmap(self, mask):
            # Put all void classes to zero
            cp = mask.clone()
            for _voidc in self.void_classes:
                mask[cp == _voidc] = self.ignore_index
            for _validc in self.valid_classes:
                mask[cp == _validc] = self.class_map[_validc]
            return mask

if __name__ == '__main__':
    dataset = cityscape_dataset(mode = 'train', interval = 1)
    for i in range(10):
        print(dataset[i])
