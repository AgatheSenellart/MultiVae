import os

import numpy as np
import pytest
from pythae.data.datasets import DatasetOutput
from torchvision.datasets import MNIST

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mnist_svhn import MnistSvhn


@pytest.mark.slow
class Test:
    @pytest.fixture
    def input_dataset_test(self, tmpdir):
        data_path = os.path.join(tmpdir, "data")
        split = "test"

        return dict(data_path=data_path, split=split)

    def test_create_dataset(self, input_dataset_test):
        try:
            mnist = MNIST(
                input_dataset_test["data_path"],
                train=(input_dataset_test["split"] == "train"),
                download=False,
            )
        except:  # If the dataset is not available don't run the test
            return

        dataset = MnistSvhn(**input_dataset_test)
        assert isinstance(dataset, MultimodalBaseDataset)
        sample = dataset[0]
        assert type(sample) == DatasetOutput
        assert len(dataset) == 50000
