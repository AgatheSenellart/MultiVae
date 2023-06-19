"""
Multimodal dataset wrapper for the MNIST labels dataset.
"""
from typing import Literal

import torch
from torch.distributions import Bernoulli
from torchvision.datasets import MNIST

from .base import DatasetOutput, MultimodalBaseDataset


class MnistLabels(MultimodalBaseDataset): # pragma: no cover
    def __init__(
        self, data_path: str, split: Literal["train", "test"] = "train",
        download=False,
        random_binarized=True,
        dtype = torch.float32
    ):
        """
        Class to wrap the MnistLabels dataset for MultiVae use. 

        Args:
            data_path (str): Where is stored the data. 
            split (Literal["train", "test"], optional): The split to use. Defaults to "train".
            download (bool, optional): If the data is not available at the given data_path, whether to download it or not.
                Defaults to False.
            random_binarized (bool, optional): Whether binarize the images using a Bernoulli distribution for each pixel.
                Defaults to True.
            dtype (_type_, optional): Type for the arrays. Defaults to torch.float32.
        """
        torchvision_dataset = MNIST(
            root=data_path, train=(split == "train"), download=download
        )

        self.images = torchvision_dataset.data.div(255).unsqueeze(1).to(dtype)
        self.labels = torchvision_dataset.targets
        self.labels_one_hot =  torch.zeros(len(self.labels), 10)
        self.labels_one_hot = self.labels_one_hot.scatter(1, self.labels.unsqueeze(1), 1)
        self.labels_one_hot = self.labels_one_hot.unsqueeze(1)
        self.random_binarized = random_binarized

    def __getitem__(self, index):
        if self.random_binarized:
            images = Bernoulli(self.images[index]).sample()
        else :
            images = self.images[index]
        return DatasetOutput(
            
            data=dict(
                images=images, labels=self.labels_one_hot[index]
            ),
            labels=self.labels[index],
        )

    def __len__(self):
        return len(self.labels)
