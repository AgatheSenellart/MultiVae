import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN

from .base import MultimodalBaseDataset
from .utils import ResampleDataset

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MnistSvhn(MultimodalBaseDataset):  # pragma: no cover
    """

    A paired MnistSvhn dataset.

    Args:
        path (str) : The path where the data is saved.
        split (str) : Either 'train' or 'test'.
        download (bool) : Whether to download the data or not. Default to True.
        data_multiplication (int) :

        **kwargs:

            transform_mnist (Transform) : a transformation to apply to MNIST. If none specified, a simple ToTensor() is applied.
            transform_svhn (Transform) : a transformation to apply to SVHN. If none specified, a simple ToTensor() is applied.

    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        download=False,
        data_multiplication=5,
        **kwargs,
    ):
        if split not in ["train", "test"]:
            raise AttributeError("Possible values for split are 'train' or 'test'")

        # Load unimodal datasets

        mnist = MNIST(data_path, train=(split == "train"), download=download)
        svhn = SVHN(data_path, split=split, download=download)

        self.data_mul = data_multiplication
        self.path_to_idx = (
            data_path + f"/mnist_svhn_idx_data_mul_{self.data_mul}/" + split
        )
        # Check if a pairing already exists and if not create one
        if not self._check_pairing_exists():
            self.create_pairing(mnist, svhn)

        i_mnist = torch.load(
            f"{self.path_to_idx}/mnist_idx.pt", weights_only=True
        )  ## !!!!WARNING!!!
        i_svhn = torch.load(
            f"{self.path_to_idx}/svhn_idx.pt", weights_only=True
        )  ## !!!!WARNING!!!

        order = np.arange(len(i_mnist))
        np.random.shuffle(
            order
        )  # shuffle the samples so that they are not ordered by labels.
        labels = mnist.targets[i_mnist][order]

        # Resample the datasets

        data_mnist = mnist.data.float().div(255).unsqueeze(1)
        data_svhn = torch.FloatTensor(svhn.data).div(255)
        mnist = ResampleDataset(
            data_mnist, lambda d, i: i_mnist[order[i]], size=len(i_mnist)
        )
        svhn = ResampleDataset(
            data_svhn, lambda d, i: i_svhn[order[i]], size=len(i_svhn)
        )
        data = dict(mnist=mnist, svhn=svhn)

        self.data_path = data_path
        super().__init__(data, labels)

    def _check_pairing_exists(self):
        if not os.path.exists(f"{self.path_to_idx}/mnist_idx.pt"):
            logger.warning("Pairing not found.")
            return False
        if not os.path.exists(f"{self.path_to_idx}/svhn_idx.pt"):
            logger.warning("Pairing not found.")
            return False
        return True

    def rand_match_on_idx(self, l1, idx1, l2, idx2, max_d=10000):
        _idx1, _idx2 = [], []
        for l in l1.unique():  # assuming both have same idxs
            l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(self.data_mul):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)

    def create_pairing(self, mnist: MNIST, svhn: SVHN, max_d=10000):
        logger.info(f"Creating indices in {self.path_to_idx}")
        # Refactor svhn labels to match mnist labels
        svhn.labels = torch.LongTensor(svhn.labels.squeeze().astype(int)) % 10
        mnist_l, mnist_li = mnist.targets.sort()
        svhn_l, svhn_li = svhn.labels.sort()
        idx1, idx2 = self.rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d
        )

        path = Path(self.path_to_idx)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(idx1, f"{self.path_to_idx}/mnist_idx.pt")
        torch.save(idx2, f"{self.path_to_idx}/svhn_idx.pt")
