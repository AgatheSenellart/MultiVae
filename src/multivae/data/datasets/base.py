from typing import Union

from numpy import ndarray
from pythae.data.datasets import Dataset, DatasetOutput
from torch import Tensor


class MultimodalBaseDataset(Dataset):
    """
    This class is the base class for datasets. A ``__getitem__`` is redefined and outputs a
    python dictionary with the keys corresponding to `data` and `labels`. You can use this
    class to define new datasets.

    If you want, you can also create your own dataset class, inheriting from
    MultimodalBaseDataset and overwriting the __getitem__ function. (Just make sure the output
    format stays the same).
    For instance:

    .. code-block:: python

        >>> from multivae.data.datasets import MultimodalBaseDataset, DatasetOutput
        >>>
        >>> class MyDataset(MultimodalBaseDataset):
        ...     def __init__(self, my_arguments):
        ...         # your code
        ...
        ...     def __getitem__(self, index):
        ...         # your code
        ...
        ...         return DatasetOutput(
        ...                     data = your_data # must be a Dict[str, Tensor],
        ...                     labels = your_labels # optional : don't add this field if you don't have labels
        ...     )


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

        if self.labels is not None:
            y = self.labels[index]
            return DatasetOutput(data=X, labels=y)
        return DatasetOutput(data=X)

    def transform_for_plotting(self, tensor, modality):
        """
        A function that to override in subclasses if you want to transform
        a tensor data for plotting. This function is called by the BaseTrainer
        to visualize generations during training and by the Visualization Module.


        For instance: if you have a 3D dimensional images, you might want
        to visualize generations during training but only 2D images can be
        logged to wandb.
        In that case, you can override this function in your dataset class. For instance with;

        .. code-block:: python

            >>> def transform_for_plotting(self, tensor, modality):
            ...     if modality == '3Dimage':
            ...         return tensor[:, 0, :, :] # select a slice
            ...     return tensor




        """

        return tensor


class IncompleteDataset(MultimodalBaseDataset):
    """
    This class is the base class for datasets with incomplete data.
    We add a field masks to indicate which data samples are available.
    This is used with models compatible with partial data.
    A ``__getitem__`` is redefined and outputs a python dictionary with the keys
    corresponding to `data` and `masks` (optionally `labels`). 
    This class should be used for any new incomplete datasets.

    If you want, you can also create your own dataset class, inheriting from
    IncompleteDataset and overwriting the __getitem__ function. (Just make sure the output
    format stays the same).

    For instance:

    .. code-block:: python

        >>> from multivae.data.datasets import IncompleteDataset, DatasetOutput
        >>>
        >>> class MyDataset(IncompleteDataset):
        ...     def __init__(self, my_arguments):
        ...         # your code
        ...
        ...     def __getitem__(self, index):
        ...         # your code
        ...         your_data = {
        ...             'modality_name_1' :  ....
        ...             'modality_name_2' : ...
        ...             }
        ...         # Warning : if 'modality_name_2' is unavailable for this index
        ...         # Artificially fill the value data['modality_name_2'] with
        ...         # a zero-tensor (or any value you want, it doesn't matter) OF THE RIGHT SHAPE.
        ...         # Otherwise MultiVae models won't work.
        ...
        ...         your_masks = { 'modality_name_1' : True,
        ...                         'modality_name_2' : False # set to False is the modality is unavailable.
        ...             }
        ...
        ...
        ...         return DatasetOutput(
        ...                     data = your_data # must be a Dict[str, Tensor],
        ...                    masks = your_masks # must be a Dict[str, 1d Tensor],
        ...                     labels = your_labels # optional : don't add this field if you don't have labels
        ...     )


    .. warning::

        If you intend to define your own IncompleteDataset subclass, please take a close look at the code
        snippet before doing so.

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
        self.check_lenght()

    def check_lenght(self):
        length = len(self.data[list(self.data)[0]])

        # check that all modalities have the same number of samples
        for m in self.data:
            if len(self.data[m]) != length or len(self.masks[m]) != length:
                raise AttributeError(
                    "The size of the provided datasets/masks doesn't correspond between modalities!"
                )
        # check that labels have the same number of samples
        if self.labels is not None:
            if len(self.labels) != length:
                raise AttributeError(
                    "The size of the provided datasets/masks doesn't correspond with the labels"
                )
        return

    def __len__(self):
        length = len(self.data[list(self.data)[0]])

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
        m = {modality: self.masks[modality][index] for modality in self.masks}

        if self.labels is not None:
            y = self.labels[index]
            return DatasetOutput(data=X, labels=y, masks=m)
        return DatasetOutput(data=X, masks=m)
