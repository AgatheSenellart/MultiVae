import os
import numpy as np
import pytest

from multivae.data.datasets.base import MultimodalBaseDataset


class Test_BaseMultiVAEConfig:
    @pytest.fixture
    def input_dataset_test(self):
        data = dict(
            mod1 = np.array([[1,2],[4,5]]),
            mod2 = np.array([[67,2,3],[1,2,3]]),
        )
        labels = np.array([0,1])
        return dict(data=data, labels=labels)

    def test_create_dataset(self, input_dataset_test):
        dataset = MultimodalBaseDataset(**input_dataset_test)

        sample = dataset[0]
        assert type(sample)==dict
        assert sample['data'] ==  dict(
            mod1 = np.array([1,2]),
            mod2 = np.array([67,2,3]),
        )
        assert sample['label'] == 0
