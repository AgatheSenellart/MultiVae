from math import ceil, floor

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


class ResampleDataset(Dataset):
    """
    Dataset which resamples a given dataset. From torchnet's ResampleDataset.
    https://tnt.readthedocs.io/en/latest/_modules/torchnet/dataset/resampledataset.html


    Args:
        dataset (Dataset): Dataset to be resampled.
        sampler (function, optional): Function used for sampling. `idx`th
            sample is returned by `dataset[sampler(dataset, idx)]`. By default
            `sampler(dataset, idx)` is the identity, simply returning `idx`.
            `sampler(dataset, idx)` must return an index in the range
            acceptable for the underlying `dataset`.
        size (int, optional): Desired size of the dataset after resampling. By
            default, the new dataset will have the same size as the underlying
            one.

    """

    def __init__(self, dataset, sampler=lambda ds, idx: idx, size=None, transform=None):
        super(ResampleDataset, self).__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.size = size
        self.transform = transform

    def __len__(self):
        return (self.size and self.size > 0) and self.size or len(self.dataset)

    def __getitem__(self, idx):
        idx = self.sampler(self.dataset, idx)

        if torch.min(idx) < 0 or torch.max(idx) >= len(self.dataset):
            raise IndexError("out of range")

        if self.transform is not None:
            return self.transform(self.dataset[idx])
        else:
            return self.dataset[idx]


def adapt_shape(data):
    """
    Adapts the shape of the data for visualization. The output dictionary contains the same data
    but resized to be of the shape (n_data, ch=3, h, w) with h and w being the largest height and
    width accross data modalities.
    """
    # First, add dimensions if some are missing and adjust the number of channels
    for m in data:
        if len(data[m].shape) == 1:  # (n_data,)
            data[m] = data[m].unsqueeze(1)
        if len(data[m].shape) == 2:  # (n_data, n)
            data[m] = data[m].unsqueeze(1)
        if len(data[m].shape) == 3:  # (n-data, n, m)
            data[m] = data[m].unsqueeze(1)
        if len(data[m].shape) == 4:
            if data[m].shape[1] == 1:
                # Add channels to have 3 channels
                data[m] = torch.cat([data[m] for _ in range(3)], dim=1)
            elif data[m].shape[1] == 2:
                n, ch, h, w = data[m].shape
                data[m] = torch.cat([data[m], torch.zeros(n, 1, h, w)], dim=1)
            else:
                data[m] = data[m][:, :3, :, :]
        else:
            raise AttributeError("Can't visualize data with more than 3 dimensions")

    h = max([data[m].shape[2] for m in data])
    w = max([data[m].shape[3] for m in data])
    for m in data:
        hm, wm = data[m].shape[2:]
        data[m] = F.pad(
            data[m],
            (
                floor((w - wm) / 2),
                ceil((w - wm) / 2),
                floor((h - hm) / 2),
                ceil((h - hm) / 2),
            ),
            mode="constant",
            value=0,
        )

    return data, (3, h, w)
