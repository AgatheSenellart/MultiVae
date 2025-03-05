import logging
import math
import os
import tempfile
from typing import Literal

import numpy as np
import torch
from pythae.data.datasets import DatasetOutput
from torchvision.datasets.utils import download_and_extract_archive

from .base import MultimodalBaseDataset

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MMNISTDataset(MultimodalBaseDataset):  # pragma: no cover
    """
    Multimodal PolyMNIST Dataset from
    'Generalized Multimodal Elbo' Sutter et al 2021.

    This dataset class has a parameter 'missing_ratio' that allows to simulate a dataset
    with missing values (Missing At Random).

    .. code-block:: python

        >>> from multivae.data.datasets import MMNISTDataset
        >>> dataset = MMNISTDataset(
        ...            data_path = 'your_data_path',
        ...            split = 'train',
        ...            download = True, #to download the dataset
        ...            missing_ratio = 0.2 # 20% of missing data
        ...        )

    """

    def __init__(
        self,
        data_path: str,
        transform=None,
        target_transform=None,
        split: Literal["train", "test"] = "train",
        download: bool = False,
        missing_ratio: float = 0,
        keep_incomplete: bool = True,
    ):
        """
        Args:
            data_path (str) : The path where to find the MMNIST folder containing the folders 'train' or 'test'.
                The data used is the one that can be downloaded from https://zenodo.org/record/4899160#.YLn0rKgzaHu
                If data_path doesn't contain the dataset and download is set to True, then the data can be downloaded
                automatically using gdown. For that, set download to True.
            transform: tranforms on colored MNIST digits.
            target_transform: transforms on labels.
            split (Literal['train', 'test']). Which part of the data to use.
            download (bool). Autorization to download the data if it is missing at the specified location.
            missing_ratio (float between 0 and 1) : To create an partially observed dataset, specify a missing ratio > 0 and <= 1.
                Default to 0  : No missing data.
            keep_incomplete (bool) : For a partially observed dataset, there are two options.
                Either keep all the samples and masks to train with incomplete data (set keep_incomplete to True)
                or only keep complete samples (keep_incomplete = False).
                Default to True.

        """

        if isinstance(data_path, str):
            data_path = os.path.expanduser(data_path)

        unimodal_datapaths = [
            os.path.join(data_path, "MMNIST", split, f"m{i}.pt") for i in range(5)
        ]
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.missing_ratio = missing_ratio
        self.keep_incomplete = keep_incomplete

        self.__check_or_download_data__(data_path, unimodal_datapaths)

        self.m0 = torch.load(unimodal_datapaths[0], weights_only=True)
        self.m1 = torch.load(unimodal_datapaths[1], weights_only=True)
        self.m2 = torch.load(unimodal_datapaths[2], weights_only=True)
        self.m3 = torch.load(unimodal_datapaths[3], weights_only=True)
        self.m4 = torch.load(unimodal_datapaths[4], weights_only=True)

        self.images_dict = {
            "m0": self.m0,
            "m1": self.m1,
            "m2": self.m2,
            "m3": self.m3,
            "m4": self.m4,
        }

        label_datapaths = os.path.join(data_path, "MMNIST", split, "labels.pt")

        self.labels = torch.load(label_datapaths, weights_only=True)

        assert self.m0.shape[0] == self.labels.shape[0]
        self.num_files = self.labels.shape[0]

        if missing_ratio > 0 and self.keep_incomplete:
            self.masks = {}
            for i in range(5):
                # randomly define the missing samples.
                self.masks[f"m{i}"] = torch.bernoulli(
                    torch.ones((self.num_files,)) * (1 - missing_ratio),
                    generator=torch.Generator().manual_seed(i),
                ).bool()

            self.masks["m0"] = torch.ones(
                (self.num_files,)
            ).bool()  # ensure there is at least one modality
            # available for all samples

            # To be sure, also erase the content of the masked samples
            for k in self.masks:
                reverse_dim_order = tuple(
                    np.arange(len(self.images_dict[k].shape))[::-1]
                )
                self.images_dict[k] = self.images_dict[k].permute(*reverse_dim_order)
                # now the batch dimension is last
                self.images_dict[k] *= self.masks[k].float()  # erase missing samples
                # put dimensions back in order
                self.images_dict[k] = self.images_dict[k].permute(*reverse_dim_order)

    def __check_or_download_data__(self, data_path, unimodal_datapaths):

        if not os.path.exists(unimodal_datapaths[0]) and self.download:
            tempdir = tempfile.mkdtemp()
            logger.info(
                f"Downloading the PolyMNIST dataset into {data_path}"
                " Along with the dataset, the classifiers and inception networks are also downloaded."
            )
            download_and_extract_archive(
                url="https://zenodo.org/record/4899160/files/PolyMNIST.zip",
                download_root=tempdir,
                extract_root=data_path,
            )

        elif not os.path.exists(unimodal_datapaths[0]) and not self.download:
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
        images_dict = {k: self.images_dict[k][index] for k in self.images_dict}
        if self.missing_ratio == 0 or not self.keep_incomplete:
            return DatasetOutput(data=images_dict, labels=self.labels[index])
        else:
            masks_dict = {k: self.masks[k][index] for k in self.masks}

            return DatasetOutput(
                data=images_dict, labels=self.labels[index], masks=masks_dict
            )

    def __len__(self):
        if self.missing_ratio == 0 or self.keep_incomplete:
            return self.num_files
        else:
            # Reduce the lenght using the proportion of complete samples
            # that corresponds to missing_ratio
            new_length = math.ceil((1 - self.missing_ratio) ** 4 * self.num_files)
            return new_length
