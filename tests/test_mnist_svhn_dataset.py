import os

import numpy as np
import pytest
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mnist_svhn import MnistSvhn


class Test:
    @pytest.fixture
    def input_dataset_test(self):
        data_path = "../data"
        split = "test"

        return dict(data_path=data_path, split=split)

    def test_create_dataset(self, input_dataset_test):
        dataset = MnistSvhn(**input_dataset_test)
        assert isinstance(dataset, MultimodalBaseDataset)
        sample = dataset[0]
        assert type(sample) == DatasetOutput
        assert len(dataset) == 50000
