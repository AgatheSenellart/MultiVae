'''
Multimodal dataset wrapper for the MNIST labels dataset.
'''
import torch
from .base import MultimodalBaseDataset, DatasetOutput
from typing import Literal
from torchvision.datasets import MNIST
from torch.distributions import Bernoulli

class BinaryMnistLabels(MultimodalBaseDataset):
    
    def __init__(self, data_path: str, split: Literal["train","test"] = "train",download=False):
        
        torchvision_dataset = MNIST(root=data_path,train=(split=="train"),download=download)
        
        self.images = torchvision_dataset.data.float().div(255).unsqueeze(1)
        self.labels = torchvision_dataset.targets
        
        
    def __getitem__(self, index):
        
        return DatasetOutput(
            data = dict(
                images = Bernoulli(self.images[index]).sample(),
                labels = self.labels[index]
            ),
            labels = self.labels[index]
        )
        
    def __len__(self):
        return len(self.labels)
    