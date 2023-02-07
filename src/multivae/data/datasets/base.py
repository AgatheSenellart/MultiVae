from collections import OrderedDict
from typing import Any, Tuple

import torch
from torch.utils.data import Dataset


from pythae.data.datasets import Dataset


class MultimodalBaseDataset(Dataset):
    """This class is the Base class for datasets.

        A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """

    def __init__(self, data : dict, labels):

        self.labels = labels.type(torch.float)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = {modality : self.data[modality][index] for modality in self.data }
        y = self.labels[index]

        return dict(data=X, labels=y)