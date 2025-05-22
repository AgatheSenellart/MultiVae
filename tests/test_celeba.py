from torch.utils.data import DataLoader

from multivae.data.datasets.celeba import CelebAttr, DatasetOutput


class TestCelebAttr:
    """Test class for CelebAttr dataset.
    This test only works locally with the CelebAttr downloaded in the ../data folder.
    """

    def test(self):
        try:
            dataset = CelebAttr("../data", "valid")
        except RuntimeError:
            return
        # Check the output of the dataset
        item = dataset[0]
        assert isinstance(item, DatasetOutput)
        assert hasattr(item, "data")
        assert hasattr(item, "labels")
        # Check the shape of the data
        assert item.data["image"].shape == (3, 64, 64)
        assert item.data["attributes"].shape == (18,)

        # Check compatibility with DataLoader
        dl = DataLoader(dataset, 12)
        batch = next(iter(dl))

        assert isinstance(batch, DatasetOutput)
        assert hasattr(batch, "data")
        assert hasattr(batch, "labels")

        assert batch.data["image"].shape == (12, 3, 64, 64)
        assert batch.data["attributes"].shape == (12, 18)
