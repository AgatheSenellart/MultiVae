from torch.utils.data import Dataset


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

    def __init__(self, dataset, sampler=lambda ds, idx: idx, size=None):
        super(ResampleDataset, self).__init__()
        self.dataset = dataset
        self.sampler = sampler
        self.size = size

    def __len__(self):
        return (self.size and self.size > 0) and self.size or len(self.dataset)

    def __getitem__(self, idx):
        idx = self.sampler(self.dataset, idx)

        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("out of range")

        return self.dataset[idx]
