from pythae.data.datasets import Dataset

class MultimodalBaseDataset(Dataset):
    """This class is the Base class for datasets.
        A ``__getitem__`` is redefined and outputs a python dictionnary
    with the keys corresponding to `data` and `labels`.
    This Class should be used for any new data sets.
    """
    def __len__(self):
        return 1
    def __getitem__(self, index):
        return 1