import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn.functional import one_hot

from .base import DatasetOutput, IncompleteDataset


def unstack_tensor(tensor, dim=0):
    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack


class MHD(IncompleteDataset):  # pragma: no cover
    """

    Dataset class for the MHD dataset introduced in the paper:
    'Leveraging hierarchy in multimodal generative models for effective
    cross-modality inference' (Vasco et al, 2021).'

    In this version of the dataset class, we add the possibility to
    simulate missingness in the data, depending on the dataclass (Missing Not At Random).
    For that, the `missing_probabilities` parameter provides probabilities of missingness for each class,
    and for each modality. For instance, the code below will define a dataset with missing samples in the
    trajectory modality, only in the classes 0,1,2, et 9.

    .. code-block:: python

        >>> from multivae.data.datasets import MHD
        >>> missing_probabilities = {
        ...     image = np.zeros(10,).float(),
        ...     audio = np.zeros(10,).float(),
        ...     trajectory = [0.1,0.3,0.4,0.,0.,0.,0.,0.,0.,0.9]
        ... }
        >>> dataset = MHD(data_path,
        ...  'train',
        ...   modalities = ['image', 'audio', 'trajectory'],
        ...   download = True,
        ...   missing_probabilities = missing_probabilities)


    Args:

        datapath (str) : Where the data is stored. It must contained the 'mhd_train.pt' file and
            'mhd_test.pt' file.
        split (Literal['train', 'test']) : Split of the data to use. Default to 'train'.
        modalities (list) :  The modalities to use among 'label', 'trajectory', 'image', 'audio'.
            By default, we use all.
        download (bool) : If the dataset is not present at the given path, wether to download it or not.
             Default to False.
        missing_probabilities (dict) : For each modality, the probabilities for each class
            to be missing in the created incomplete dataset. By default, we use no missing data.
        seed (int) : default to 0. You can change the seed to create a different incomplete dataset.


    """

    def __init__(
        self,
        datapath: str,
        split="train",
        modalities: list = ["label", "audio", "trajectory", "image"],
        download=False,
        missing_probabilities=dict(
            label=[0.0] * 10, audio=[0.0] * 10, trajectory=[0.0] * 10, image=[0.0] * 10
        ),
        seed=0,
    ):
        self.data_file = os.path.join(datapath, f"mhd_{split}.pt")
        self.modalities = modalities
        if not os.path.exists(self.data_file):
            if not download:
                raise RuntimeError(
                    f"Dataset not found at path {datapath} and download is set to False. "
                    "Please change the path or set download to True"
                )
            else:
                try:
                    self.__download__(split, datapath)

                except:
                    raise RuntimeError(
                        "gdown must be installed to download the dataset automatically."
                        "Install gdown with "
                        ' "pip install gdown" or download the dataset manually at the following url'
                        "train : https://docs.google.com/uc?export=download&id=1Tj1i-hXA0INQpU0jmuTMO4IwfDoGD2oV"
                        "test : https://docs.google.com/uc?export=download&id=1qiEjFNCFn1ws383pKmY3zJtm4JDymOU6"
                    )

        (
            self._s_data,
            self._i_data,
            self._t_data,
            self._a_data,
            self._traj_normalization,
            self._audio_normalization,
        ) = torch.load(self.data_file)

        self.data = dict()
        if "image" in modalities:
            self.data["image"] = self._i_data
        if "label" in modalities:
            self.data["label"] = one_hot(self._s_data, num_classes=10).float()
        if "trajectory" in modalities:
            self.data["trajectory"] = self._t_data
        if "audio" in modalities:
            self.data["audio"] = self._a_data

        self.labels = self._s_data
        self.n_data = len(self._s_data)
        self.is_incomplete = (
            sum([sum(missing_probabilities[s]) for s in missing_probabilities]) != 0
        )

        if self.is_incomplete:
            # generate the masks
            self.masks = {}
            for i, mod in enumerate(self.data):
                # randomly define the missing samples.
                p = 1 - torch.tensor(missing_probabilities[mod])[self._s_data]
                self.masks[mod] = torch.bernoulli(
                    p, generator=torch.Generator().manual_seed(seed + i)
                ).bool()

            # To be sure, also erase the content of the masked samples
            for k in self.masks:
                reverse_dim_order = tuple(np.arange(len(self.data[k].shape))[::-1])
                self.data[k] = self.data[k].permute(*reverse_dim_order).float()
                # now the batch dimension is last
                self.data[k] *= self.masks[k].float()  # erase missing samples
                # put dimensions back in order
                self.data[k] = self.data[k].permute(*reverse_dim_order)

    def __download__(self, split, datapath):  # pragram : no cover
        import gdown

        if not os.path.exists(datapath):
            os.makedirs(Path(datapath), exist_ok=True)

        if split == "train":
            gdown.download(
                "https://docs.google.com/uc?export=download&id=1Tj1i-hXA0INQpU0jmuTMO4IwfDoGD2oV",
                output=os.path.join(datapath, f"mhd_{split}.pt"),
            )
        else:
            gdown.download(
                "https://docs.google.com/uc?export=download&id=1qiEjFNCFn1ws383pKmY3zJtm4JDymOU6",
                output=os.path.join(datapath, f"mhd_{split}.pt"),
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (t_data, m_data, f_data)
        """

        data = {s: self.data[s][index] for s in self.data}

        if "audio" in data:
            # Audio modality is a 3x32x32 representation, need to unstack!
            audio = unstack_tensor(data["audio"]).unsqueeze(0)
            data["audio"] = audio.permute(0, 2, 1)

        if not self.is_incomplete:
            return DatasetOutput(data=data, labels=self._s_data[index])
        else:
            masks = {s: self.masks[s][index] for s in self.data}
            return DatasetOutput(data=data, labels=self._s_data[index], masks=masks)

    def __len__(self):
        return self.n_data

    def get_audio_normalization(self):
        return self._audio_normalization

    def get_traj_normalization(self):
        return self._traj_normalization
