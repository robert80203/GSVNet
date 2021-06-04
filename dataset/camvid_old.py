from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision.transforms as standard_transforms
#from config import get_config
from torchvision.datasets.folder import default_loader

#CamvidConfig = get_config('camvid')

class camvid_dataset(Dataset):
    def __init__(self, data_path, mode = 'train', interval = 0, label_transform = standard_transforms.ToTensor(), img_transform = standard_transforms.ToTensor(), bi_direction = False):
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
        self.valid_classes = [0,1,2,3,4,5,6,7,8,9,10] # we have 19 classes + 1 void class
        self.ignore_index = 250
        self.data_path = data_path
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        if mode == 'train' :
            file_path = "./camvid/train.txt"#"./dataset/camvid/train.txt"
        elif mode == 'val':
            file_path = "./dataset/camvid/val.txt"
        elif mode == 'test':
            file_path = "./dataset/camvid/test.txt"
        
        self.file_list = []
        with open(file_path, "r") as f :
            for line in f:
                string = line.split(" ")[0]
                label_path = line.split(" ")[1][:-1].split("/")[3]
                file_name = string[:-16]
                #back_address = string[-16:]
                back_address = string[-4:]
                frame_idx = file_name[-6:]
                file_name = file_name[:-6]
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
                                (self.data_path, file_name + "{:06d}".format(frame_num) + back_address))   
            img = self.img_transform(img)
            imgs.append((img))
            idxes.append(frame_num)

        if self.mode != 'test':
            label = default_loader(os.path.join
                                (self.data_path, self.mode, label_path.split("_")[0], label_path)
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
    dataset = camvid_dataset(data_path='/work/daisy91530/camvid/', mode = 'train', interval = 1)
    for i in range(10):
        print(dataset[i])
