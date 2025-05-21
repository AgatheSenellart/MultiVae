import numpy as np
import pytest
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset


class TestMultimodalDataset:
    @pytest.fixture
    def input_dataset_test(self):
        """Create data for testing the MultimodalBaseDataset class"""
        data = dict(
            mod1=np.array([[1, 2], [4, 5]]),
            mod2=np.array([[67, 2, 3], [1, 2, 3]]),
        )
        labels = np.array([0, 1])
        return dict(data=data, labels=labels)

    def test_create_dataset(self, input_dataset_test):
        """Test the init and __getitem__ function of MultimodalBaseDataset."""
        dataset = MultimodalBaseDataset(**input_dataset_test)

        sample = dataset[0]
        assert isinstance(sample, DatasetOutput)
        assert np.all(sample["data"]["mod1"] == np.array([1, 2]))
        assert np.all(sample["data"]["mod2"] == np.array([67, 2, 3]))
        assert sample["labels"] == 0
