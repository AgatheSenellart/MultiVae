import os

import numpy as np
import pytest
import torch
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mmnist import MMNISTDataset


class Test:
    @pytest.fixture
    def input_dataset_test(self, tmpdir):
        data_path = os.path.join(tmpdir, "data")
        split = "test"

        return dict(data_path=data_path, split=split)

    def test_create_dataset(self, input_dataset_test):
        if not os.path.exists(input_dataset_test["data_path"]):
            Warning("The MMNIST dataset is not available")
            return
        else:
            dataset = MMNISTDataset(**input_dataset_test)
            assert isinstance(dataset, MultimodalBaseDataset)
            sample = dataset[0]
            assert type(sample) == DatasetOutput
            assert isinstance(sample.data["m0"], torch.Tensor)
            assert torch.min(sample.data["m0"]) >= 0
            assert torch.max(sample.data["m0"]) <= 1

            assert torch.min(sample.data["m1"]) >= 0
            assert torch.max(sample.data["m1"]) <= 1

            assert torch.min(sample.data["m2"]) >= 0
            assert torch.max(sample.data["m2"]) <= 1
            assert torch.min(sample.data["m3"]) >= 0
            assert torch.max(sample.data["m3"]) <= 1

            assert torch.min(sample.data["m4"]) >= 0
            assert torch.max(sample.data["m4"]) <= 1

            assert sample.data["m0"].size() == torch.Size([3, 28, 28])
            assert len(dataset) == 10000
