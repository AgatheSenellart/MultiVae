from typing import Union
from numpy import ndarray

from torch import Tensor
from .base import MultimodalBaseDataset
import os
from torchvision.transforms import CenterCrop, ToTensor, Compose
from pathlib import Path
from PIL.Image import Image, open
import torch

class CelebAMasks(MultimodalBaseDataset):
    
    
    def __init__(self,data_folder = '../data/'):
        
        self.data_folder = Path(data_folder)
        
        if not (self.data_folder / 'CelebAMask-HQ').exists():
            raise NotADirectoryError("Please provide the path to the folder containing the "
                                     "CelebAMask-HQ folder downloaded from "
                                     "http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html.")
            
        self.img_transform = Compose([ToTensor(),CenterCrop(size=torch.Size([512,512]))])
        self.mask_transform = Compose([ToTensor(),CenterCrop(size=torch.Size([512,512]))])

        self.image_folder = self.data_folder / 'CelebAMask-HQ' / 'CelebA-HQ-img'
        self.masks_folder = self.data_folder / 'CelebAMask-HQ' / 'CelebAMask-HQ-mask-anno'
        self.parts_of_masks = ['r_eye','l_eye']
    
    
    def __getitem__(self, index):
        
        img_path = self.image_folder / (str(index) + '.jpg')
        img = open(img_path)
        img = self.img_transform(img)
        
        # get the masks
        index_mask_path = index // 2000
        mask_pref = '0'*(5-len(str(index))) + str(index) 
        mask = 0
        for suff in self.parts_of_masks:
            part_mask_path = mask_pref +'_'+ suff + '.png'
            part_mask = open(self.masks_folder / str(index_mask_path) / part_mask_path)
            part_mask = self.mask_transform(part_mask)
            mask = part_mask + mask

        return img,mask
    
    def __len__(self):
        return 30000