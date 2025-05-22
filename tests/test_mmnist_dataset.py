import os
import warnings

import pytest
import torch
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mmnist import MMNISTDataset


class TestMMNISDataset:
    """Test class for MMNIST dataset.
    This test only works locally with the MMNIST downloaded in the ../data folder.
    """

    @pytest.fixture(params=[0.2, 0])
    def input_dataset_test(self, request):
        data_path = "../data"
        split = "test"
        missing_ratio = request.param

        return dict(data_path=data_path, split=split, missing_ratio=missing_ratio)

    def test_create_dataset(self, input_dataset_test):
        if os.path.exists(
            os.path.join(input_dataset_test["data_path"], "MMNIST", "test", "m0.pt")
        ):
            dataset = MMNISTDataset(**input_dataset_test)
            assert isinstance(dataset, MultimodalBaseDataset)
            sample = dataset[0]
            assert isinstance(sample, DatasetOutput)
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
            sample = dataset[:100]
            if input_dataset_test["missing_ratio"] > 0:
                assert hasattr(sample, "masks")
                assert torch.all(sample.masks["m0"] == 1)
                for m in ["m1", "m2", "m3", "m4"]:
                    assert torch.all(
                        sample.data[m][(1 - sample.masks[m].int()).bool()] == 0
                    )
                    assert not torch.all(sample.data[m][sample.masks[m]] == 0)

                    assert torch.all(sample.masks["m0"])
                    assert not torch.all(sample.masks["m0"] == sample.masks["m1"])
        else:
            warnings.warn(
                message="PolyMNIST dataset is not downloaded at provided path and therefore"
                + "has not been tested"
            )
