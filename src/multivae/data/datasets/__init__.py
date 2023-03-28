"""
In this section, you will find all the `built-in` datasets that are currently implemented in `multivae` library
"""

from .base import MultimodalBaseDataset, IncompleteDataset
from .mnist_svhn import MnistSvhn

__all__ = ["MultimodalBaseDataset", "MnistSvhn", "IncompleteDataset"]
