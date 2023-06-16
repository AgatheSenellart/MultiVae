from typing import Union

from numpy import ndarray
from pythae.data.datasets import Dataset, DatasetOutput
from torch import Tensor


class MultimodalBaseDataset(Dataset):
    """This class is the Base class for datasets. A ``__getitem__`` is redefined and outputs a
    python dictionnary with the keys corresponding to `data` and `labels`. This Class should be
    used for any new data sets.

    Args:
        data (dict) : A dictionary containing the modalities' name and a tensor or numpy array for each modality.
        labels (Union[torch.Tensor, numpy.ndarray]) : A torch.Tensor or numpy.ndarray instance containing the labels.
    """

    def __init__(self, data: dict, labels: Union[Tensor, ndarray] = None):
        self.labels = labels
        self.data = data

    def __len__(self):
        length = len(self.data[list(self.data)[0]])
        for m in self.data:
            if len(self.data[m]) != length:
                raise AttributeError(
                    "The size of the provided datasets doesn't correspond between modalities!"
                )

        return length

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = {modality: self.data[modality][index] for modality in self.data}
        y = self.labels[index] if self.labels is not None else None

        return DatasetOutput(data=X, labels=y)


class IncompleteDataset(MultimodalBaseDataset):
    """This class is the Base class for datasets with incomplete data.
    We add a field masks to indicate which data samples are available.
    This is used with models compatible with weakly supervised learning such as
    the MVAE. A ``__getitem__`` is redefined and outputs a python dictionnary with the keys
    corresponding to `data` and `labels`. This Class should be used for any new data sets.

    Args:
        data (dict[str, torch.Tensor]) : A dictionary containing the modalities' name and a tensor or numpy array for each modality.
        masks (dict[str, torch.Tensor]) : A dictionary containing the modalities'name and a boolean tensor of the same lenght
            as the data tensor in the data dictionary. For each modality, the mask tensor indicates if a sample
            is available. The unavailable samples are assumed to have been filled by random/or zeros values in
            the data dictionary.
        labels (Union[Tensor, numpy.ndarray]) : A torch.Tensor or numpy.ndarray instance containing the labels.

    """

    def __init__(self, data: dict, masks: dict, labels: Tensor = None) -> None:
        
        self.data = data
        self.masks = masks
        self.labels = labels

    def __len__(self):
        length = len(self.data[list(self.data)[0]])
        for m in self.data:
            if len(self.data[m]) != length or len(self.masks[m]) != length:
                raise AttributeError(
                    "The size of the provided datasets/masks doesn't correspond between modalities!"
                )

        return length

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index (int): The index of the data in the Dataset

        Returns:
            (dict): A dictionnary with the keys 'data' and 'labels' and corresponding
            torch.Tensor
        """
        # Select sample
        X = {modality: self.data[modality][index] for modality in self.data}
        y = self.labels[index] if self.labels is not None else None
        m = {modality: self.masks[modality][index] for modality in self.masks}

        return DatasetOutput(data=X, labels=y, masks=m)
