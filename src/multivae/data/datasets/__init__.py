"""
In this section, you will find all the `built-in` datasets that are currently implemented in `multivae` library
"""

from .base import DatasetOutput, IncompleteDataset, MultimodalBaseDataset
from .cub import CUB
from .mhd import MHD
from .mmnist import MMNISTDataset
from .mnist_svhn import MnistSvhn

__all__ = [
    "MultimodalBaseDataset",
    "MnistSvhn",
    "IncompleteDataset",
    "DatasetOutput",
    "CUB",
    "MHD",
    "MMNISTDataset",
]
