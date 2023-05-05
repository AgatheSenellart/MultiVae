"""
Multimodal dataset wrapper for the MNIST labels dataset.
"""
from typing import Literal

from torch.distributions import Bernoulli
from torchvision.datasets import MNIST

from .base import DatasetOutput, MultimodalBaseDataset


class BinaryMnistLabels(MultimodalBaseDataset):
    def __init__(
        self, data_path: str, split: Literal["train", "test"] = "train",
        download=False,
        random_binarized=True,
    ):
        torchvision_dataset = MNIST(
            root=data_path, train=(split == "train"), download=download
        )

        self.images = torchvision_dataset.data.float().div(255).unsqueeze(1)
        self.labels = torchvision_dataset.targets
        self.random_binarized = random_binarized

    def __getitem__(self, index):
        if self.random_binarized:
            images = Bernoulli(self.images[index]).sample()
        else :
            images = self.images[index]
        return DatasetOutput(
            
            data=dict(
                images=images, labels=self.labels[index]
            ),
            labels=self.labels[index],
        )

    def __len__(self):
        return len(self.labels)
