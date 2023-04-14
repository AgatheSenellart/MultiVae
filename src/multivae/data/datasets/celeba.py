from typing import Literal

from torchvision import transforms
from torchvision.datasets import CelebA

from .base import DatasetOutput, MultimodalBaseDataset


class CelebAttr(MultimodalBaseDataset):
    
    def __init__(self, root : str, split :str, transform = None, target_transform = None,attributes : Literal['18','40']='18',download = False): 
        
        self.root = root
        
        if transform is None:
            transform = transforms.Compose([transforms.Resize(64),
                                          transforms.CenterCrop(64),
                                          transforms.ToTensor()])
        self.transform = transform
        
        self.torchvision_dataset = CelebA(root=root,
                                          split=split,
                                          target_type='attr',
                                          transform = transform,
                                          download=download
                                          )
        
        if attributes == '18' :
            self.attributes_to_keep = [4, 5, 8, 9, 11, 12, 15, 17, 18, 20, 21, 22, 26, 28, 31, 32, 33, 35]
        else:
            self.attributes_to_keep = range(40)
            
        self.attr_to_idx  = {'Sideburns': 30, 'Black_Hair': 8, 'Wavy_Hair': 33, 'Young': 39, 'Heavy_Makeup': 18, 
                    'Blond_Hair': 9, 'Attractive': 2, '5_o_Clock_Shadow': 0, 'Wearing_Necktie': 38, 
                    'Blurry': 10, 'Double_Chin': 14, 'Brown_Hair': 11, 'Mouth_Slightly_Open': 21, 
                    'Goatee': 16, 'Bald': 4, 'Pointy_Nose': 27, 'Gray_Hair': 17, 'Pale_Skin': 26, 
                    'Arched_Eyebrows': 1, 'Wearing_Hat': 35, 'Receding_Hairline': 28, 'Straight_Hair': 32, 
                    'Big_Nose': 7, 'Rosy_Cheeks': 29, 'Oval_Face': 25, 'Bangs': 5, 'Male': 20, 'Mustache': 22, 
                    'High_Cheekbones': 19, 'No_Beard': 24, 'Eyeglasses': 15, 'Bags_Under_Eyes': 3, 
                    'Wearing_Necklace': 37, 'Wearing_Lipstick': 36, 'Big_Lips': 6, 'Narrow_Eyes': 23, 
                    'Chubby': 13, 'Smiling': 31, 'Bushy_Eyebrows': 12, 'Wearing_Earrings': 34}
        
        self.idx_to_attr = {v:k for k, v in self.attr_to_idx.items()}

    def __getitem__(self, index):
        
        img,target = self.torchvision_dataset[index]
        
        return DatasetOutput(
            data = dict(image = img,
                        attributes = target[self.attributes_to_keep]),
            labels = target[self.attributes_to_keep]
            
        )
        
    def __len__(self):
        return self.torchvision_dataset.__len__()
        
