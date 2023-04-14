import pytest
from torch.utils.data import DataLoader

from multivae.data.datasets.celeba import CelebAttr, DatasetOutput


class Test:
    def test(self):
        try:
            dataset = CelebAttr("../data", "valid")
        except:
            return
        item = dataset[0]
        assert isinstance(item, DatasetOutput)
        assert hasattr(item, "data")
        assert hasattr(item, "labels")

        assert item.data["image"].shape == (3, 64, 64)
        assert item.data["attributes"].shape == (18,)

        dl = DataLoader(dataset, 12)
        batch = next(iter(dl))

        assert isinstance(batch, DatasetOutput)
        assert hasattr(batch, "data")
        assert hasattr(batch, "labels")

        assert batch.data["image"].shape == (12, 3, 64, 64)
        assert batch.data["attributes"].shape == (12, 18)
