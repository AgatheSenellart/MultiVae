from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch
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
            
            return(self.transform(self.dataset[idx]))
        else :
            return self.dataset[idx]


def save_all_images(data: dict, dir='',suffix=''):
    for m in data:
        save_image(data[m],dir + m + suffix + '.png')