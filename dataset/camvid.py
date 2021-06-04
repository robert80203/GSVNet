import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision.transforms as standard_transforms
#from config import get_config
from torchvision.datasets.folder import default_loader

#CamvidConfig = get_config('camvid')

class camvid_dataset(Dataset):
    def __init__(self, config, mode = 'train', interval = 0, label_transform = standard_transforms.ToTensor(), img_transform = standard_transforms.ToTensor(), bi_direction = False):
        self.mode = mode
        if interval < 0 :
            self.reverse = True
            interval = -interval
        else:
            self.reverse = False
        self.interval = interval
        self.img_transform = img_transform
        self.label_transform = label_transform

        self.bi_direction = bi_direction

        #dataset_config=CamvidConfig
        self.valid_classes = [0,1,2,3,4,5,6,7,8,9,10]
        self.ignore_index = config.data_path
        self.data_path = config.data_path
        self.train_prefix = 'frames/'
        self.eval_prefix = 'label_id/'

        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))

        if mode == 'train' :
            file_path = "./dataset/camvid/train.txt"
        elif mode == 'val':
            file_path = "./dataset/camvid/val.txt"
        elif mode == 'test':
            file_path = "./dataset/camvid/test.txt"
        
        self.file_list = []
        with open(file_path, "r") as f :
            for line in f:
                    string = line.split(" ")[0]
                    label_path = line.split(" ")[1][:-1].split("/")[1]
                    file_name = string[4:-4]
                    back_address = string[-4:]
                    frame_idx = file_name[-5:]
                    file_name = file_name[:-5]
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
                                (self.data_path+self.train_prefix+file_name.split("_")[0],
                                file_name + "{:05d}".format(frame_num) + back_address))   
            
            img = np.array(img).astype(np.float32)
            img = img/255.0
            img = img - np.array([0.41189489566336, 0.4251328133025, 0.4326707089857])
            img = img / np.array([0.27413549931506, 0.28506257482912, 0.28284674400252])
            img = np.ascontiguousarray(img[ :, :, :],
                                          dtype=np.float32).transpose(2,0,1)
            img = torch.tensor(img)
            imgs.append((img))
            idxes.append(frame_num)

        if self.mode != 'test':
            label = default_loader(os.path.join
                                (self.data_path+self.eval_prefix, label_path)
                                )
            label = self.encode_segmap(torch.tensor(np.array(label)[:,:,0]))
            if self.bi_direction:
                return imgs, label, idxes
            return imgs, label
        else:
            return imgs, label_path
    def encode_segmap(self, mask):
        # Put all void classes to zero
        cp = mask.clone()
        mask.fill_(self.ignore_index)
        for _validc in self.valid_classes:
            mask[cp == _validc] = _validc
        return mask