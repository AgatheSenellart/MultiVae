"""
In this section, you will find all the `built-in` datasets that are currently implemented in `multivae` library
"""

from .base import IncompleteDataset, MultimodalBaseDataset, DatasetOutput
from .mnist_svhn import MnistSvhn
from .cub import CUB

__all__ = ["MultimodalBaseDataset", "MnistSvhn", "IncompleteDataset", "DatasetOutput", "CUB"]


