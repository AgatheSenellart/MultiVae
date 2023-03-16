import os

import numpy as np
import pytest
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mmnist import MMNISTDataset
import torch


class Test:
    @pytest.fixture
    def input_dataset_test(self):
        data_path = "../../../data/MMNIST"
        split = "test"

        return dict(data_path=data_path, split=split)

    def test_create_dataset(self, input_dataset_test):
        dataset = MMNISTDataset(**input_dataset_test)
        assert isinstance(dataset, MultimodalBaseDataset)
        sample = dataset[0]
        assert type(sample) == DatasetOutput
        assert isinstance(sample.data['m0'],torch.Tensor)
        assert torch.max(sample.data['m0'])==1
        assert sample.data['m0'].size() == torch.Size([3,28,28])
        assert len(dataset) == 10000
