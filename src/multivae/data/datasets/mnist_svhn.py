import os
from pathlib import Path
from typing import Union

import torch
from torchvision.datasets import MNIST, SVHN

from .base import MultimodalBaseDataset
from .utils import ResampleDataset


class MnistSvhn(MultimodalBaseDataset):
    """

    A paired MnistSvhn dataset.

    Args:
        path (str) : The path where the data is saved.
        split (str) : Either 'train' or 'test'.
        download (bool) : Whether to download the data or not. Default to True.

        **kwargs:

            transform_mnist (Transform) : a transformation to apply to MNIST. If none specified, a simple ToTensor() is applied.
            transform_svhn (Transform) : a transformation to apply to SVHN. If none specified, a simple ToTensor() is applied.

    """

    def __init__(
        self,
        data_path: Union[str, Path] = "../data/",
        split: str = "train",
        download=True,
        **kwargs,
    ):
        if split not in ["train", "test"]:
            raise AttributeError("Possible values for split are 'train' or 'test'")

        # Load unimodal datasets

        mnist = MNIST(data_path, train=(split == "train"), download=download)
        svhn = SVHN(data_path, split=split, download=download)

        # Check if a pairing already exists and if not create one
        if not self._check_pairing_exists(data_path, split):
            self.create_pairing(mnist, svhn, data_path)

        i_mnist = torch.load(data_path + "/mnist_svhn_idx/" + split + "/mnist_idx.pt")
        i_svhn = torch.load(data_path + "/mnist_svhn_idx/" + split + "/svhn_idx.pt")

        labels = mnist.targets[i_mnist]

        # Resample the datasets
        data_mnist = (mnist.data / 255).unsqueeze(1)
        data_svhn = torch.FloatTensor(svhn.data) / 255
        print(data_mnist.shape)
        print(data_svhn.shape)
        mnist = ResampleDataset(data_mnist, lambda d, i: i_mnist[i], size=len(i_mnist))
        svhn = ResampleDataset(data_svhn, lambda d, i: i_svhn[i], size=len(i_svhn))
        data = dict(mnist=mnist, svhn=svhn)

        self.data_path = data_path
        super().__init__(data, labels)

    def _check_pairing_exists(self, data_path, split):
        if not os.path.exists(data_path + "/mnist_svhn_idx/" + split + "/mnist_idx.pt"):
            return False
        if not os.path.exists(data_path + "/mnist_svhn_idx/" + split + "/svhn_idx.pt"):
            return False
        return True

    def rand_match_on_idx(self, l1, idx1, l2, idx2, max_d=10000, dm=10):
        _idx1, _idx2 = [], []
        for l in l1.unique():  # assuming both have same idxs
            l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)

    def create_pairing(
        self, mnist: MNIST, svhn: SVHN, data_path: str, max_d=10000, dm=5
    ):
        split = svhn.split
        # Refactor svhn labels to match mnist labels
        svhn.labels = torch.LongTensor(svhn.labels.squeeze().astype(int)) % 10
        mnist_l, mnist_li = mnist.targets.sort()
        svhn_l, svhn_li = svhn.labels.sort()
        idx1, idx2 = self.rand_match_on_idx(
            mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm
        )

        path = Path(data_path + "/mnist_svhn_idx/" + split)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(idx1, str(path) + "/mnist_idx.pt")
        torch.save(idx2, str(path) + "/svhn_idx.pt")
