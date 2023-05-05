import os

import numpy as np
import pytest
import torch
from pythae.data.datasets import DatasetOutput

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.datasets.mmnist import MMNISTDataset


class Test:
    @pytest.fixture(
        params=[0.2,0]
    )
    def input_dataset_test(self, request):
        data_path = "~/scratch/data/MMNIST"
        split = "test"
        missing_ratio = request.param

        return dict(data_path=data_path, split=split,
                    missing_ratio=missing_ratio)

    def test_create_dataset(self, input_dataset_test):
            dataset = MMNISTDataset(**input_dataset_test)
            print(input_dataset_test['missing_ratio'])
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
            sample = dataset[:100]
            if input_dataset_test['missing_ratio'] >0:

                assert hasattr(sample, 'masks')
                assert torch.all(sample.masks["m0"] == 1)
                for m in ['m1','m2','m3','m4']:                
                    assert torch.all(sample.data[m][(1-sample.masks[m].int()).bool()] == 0)
                    assert not torch.all(sample.data[m][sample.masks[m]] == 0)
                    
                    assert torch.all(sample.masks['m0'] == True)
                    assert not torch.all(sample.masks['m0'] == sample.masks['m1'])