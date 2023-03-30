import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from pythae.data.datasets import Dataset, DatasetOutput
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from .base import MultimodalBaseDataset


class MMNISTDataset(MultimodalBaseDataset):
    """
    Multimodal MMNIST Dataset to load the Polymnist Dataset from 
    'Generalized Multimodal Elbo' Sutter et al 2021. 

    """

    def __init__(self, data_path, transform=None, target_transform=None, split="train", download=False):
        """
        Args: unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                correspond by index. Therefore the numbers of samples of all datapaths should match.
            transform: tranforms on colored MNIST digits.
            target_transform: transforms on labels.
        """
        unimodal_datapaths = [data_path + "/" + split + f"/m{i}.pt" for i in range(5)]
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.__check_or_download_data__(data_path, unimodal_datapaths)

        self.m0 = torch.load(unimodal_datapaths[0])
        self.m1 = torch.load(unimodal_datapaths[1])
        self.m2 = torch.load(unimodal_datapaths[2])
        self.m3 = torch.load(unimodal_datapaths[3])
        self.m4 = torch.load(unimodal_datapaths[4])

        label_datapaths = data_path + "/" + split + "/" + "labels.pt"

        self.labels = torch.load(label_datapaths)

        assert self.m0.shape[0] == self.labels.shape[0]
        self.num_files = self.labels.shape[0]
        
    def __check_or_download_data__(self, data_path, unimodal_datapaths):
        # TODO : test this function
        for i in range(5):
            if not os.path.exists(unimodal_datapaths[i]) and self.download:
                try :
                    import zipfile

                    import gdown
                    gdown.download(id='1N0v31KOgZgfkSqSiPdBKAgWIkKZIzAWb', output=data_path)
                    with zipfile.ZipFile(data_path + '/PolyMNIST.zip') as zip_ref:
                        zip_ref.extractall(data_path)
                except:
                    raise AttributeError('The PolyMNIST dataset is not available at the'
                                         ' given datapath and gdown is not installed to download it.'
                                         'Install gdown with `pip install gdown` or place the dataset'
                                         ' in the data_path folder.')
            elif not os.path.exists(unimodal_datapaths[i]) and not self.download:
                raise AttributeError('The PolyMNIST dataset is not available at the'
                                         ' given datapath and download is set to False.'
                                         'Set download to True or place the dataset'
                                         ' in the data_path folder.')


    
    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        images_dict = {
            "m0": self.m0[index],
            "m1": self.m1[index],
            "m2": self.m2[index],
            "m3": self.m3[index],
            "m4": self.m4[index],
        }

        return DatasetOutput(data=images_dict, labels=self.labels[index])

    def __len__(self):
        return self.num_files



