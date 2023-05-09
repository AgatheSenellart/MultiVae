import os

import numpy as np
import torch
from PIL import Image
from pythae.data.datasets import DatasetOutput

from .base import IncompleteDataset


class MMNISTDataset(IncompleteDataset):
    """
    Multimodal MMNIST Dataset to load the Polymnist Dataset from
    'Generalized Multimodal Elbo' Sutter et al 2021.

    """

    def __init__(
        self,
        data_path,
        transform=None,
        target_transform=None,
        split="train",
        download=False,
        missing_ratio = 0
    ):
        """
        Args: 
            unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                correspond by index. Therefore the numbers of samples of all datapaths should match.
            transform: tranforms on colored MNIST digits.
            target_transform: transforms on labels.
            split (Literal['train', 'test']). Which part of the data to use.
            download (bool). Autorization to download the data if it is missing at the specified location.
            missing_ratio (float between 0 and 1) : To create an partially observed dataset, specify a missing ratio > 0 and <= 1. 
                Default to 0  : No missing data. 
            

        """

        if isinstance(data_path, str):
            data_path = os.path.expanduser(data_path)

        unimodal_datapaths = [data_path + "/" + split + f"/m{i}.pt" for i in range(5)]
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.missing_ratio=missing_ratio

        self.__check_or_download_data__(data_path, unimodal_datapaths)

        self.m0 = torch.load(unimodal_datapaths[0])
        self.m1 = torch.load(unimodal_datapaths[1])
        self.m2 = torch.load(unimodal_datapaths[2])
        self.m3 = torch.load(unimodal_datapaths[3])
        self.m4 = torch.load(unimodal_datapaths[4])
        
        self.images_dict = {
            "m0": self.m0,
            "m1": self.m1,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
        }

        label_datapaths = data_path + "/" + split + "/" + "labels.pt"

        self.labels = torch.load(label_datapaths)

        assert self.m0.shape[0] == self.labels.shape[0]
        self.num_files = self.labels.shape[0]
        
        if missing_ratio > 0 :
            self.masks = {}
            for i in range(5):
                # randomly define the missing samples. 
                self.masks[f'm{i}'] = torch.bernoulli(torch.ones((self.num_files,))*(1-missing_ratio)).bool()
                print(self.masks[f'm{i}'])
            self.masks['m0']=torch.ones((self.num_files,)) # ensure there is at least one modality
                                                           # available for all samples
            
            # To be sure, also erase the content of the masked samples
            for k in self.masks:
                reverse_dim_order = tuple(np.arange(len(self.images_dict[k].shape))[::-1])
                self.images_dict[k] = self.images_dict[k].permute(*reverse_dim_order)
                # now the batch dimension is last
                self.images_dict[k] *= self.masks[k].float() # erase missing samples
                # put dimensions back in order
                self.images_dict[k] = self.images_dict[k].permute(*reverse_dim_order)
                
            
            
    def __check_or_download_data__(self, data_path, unimodal_datapaths):
        # TODO : test this function
        for i in range(5):
            if not os.path.exists(unimodal_datapaths[i]) and self.download:
                try:
                    import zipfile

                    import gdown

                    gdown.download(
                        id="1N0v31KOgZgfkSqSiPdBKAgWIkKZIzAWb", output=data_path
                    )
                    with zipfile.ZipFile(data_path + "/PolyMNIST.zip") as zip_ref:
                        zip_ref.extractall(data_path)
                except:
                    raise AttributeError(
                        "The PolyMNIST dataset is not available at the"
                        " given datapath and gdown is not installed to download it."
                        "Install gdown with `pip install gdown` or place the dataset"
                        " in the data_path folder."
                    )
            elif not os.path.exists(unimodal_datapaths[i]) and not self.download:
                raise AttributeError(
                    "The PolyMNIST dataset is not available at the"
                    " given datapath and download is set to False."
                    "Set download to True or place the dataset"
                    " in the data_path folder."
                )

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        images_dict = { k: self.images_dict[k][index] for k in self.images_dict
        }
        if self.missing_ratio == 0:
            return DatasetOutput(data=images_dict, labels=self.labels[index])
        else :
            masks_dict = {k : self.masks[k][index] for k in self.masks}
            
            # To be completely sure : replace masked samples with zero values
            
            return DatasetOutput(
                data = images_dict,
                labels = self.labels[index],
                masks = masks_dict
            )   

        
    def __len__(self):
        return self.num_files




