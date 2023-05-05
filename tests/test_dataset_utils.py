import os

import numpy as np
import pytest
import torch

from multivae.data.datasets import MultimodalBaseDataset
from multivae.data.datasets.utils import ResampleDataset, adapt_shape


@pytest.fixture
def dummy_data():
    return dict(
        mod1=torch.randn(6, 1, 28, 28),
        mod2=torch.randn(6, 3, 28),
        mod3=torch.randn(6, 3, 28, 28),
        mod4=torch.randn(6, 2, 28, 28),
        mod5=torch.randn(6, 20, 28, 28),
        mod6=torch.randn(6, 28),
    )


class TestUtils:
    def test_adapt_shape(self, dummy_data):
        out, _ = adapt_shape(dummy_data)
        assert out.keys() == out.keys()
        assert all([out[k].shape[1] == 3 for k in dummy_data.keys()])

        new_dummy_data = dict(wrong_mod=torch.randn(10, 2, 2, 4, 2))
        with pytest.raises(AttributeError):
            _ = adapt_shape(new_dummy_data)

    def test_resample_dataset(self, dummy_data):
        dataset = ResampleDataset(
            MultimodalBaseDataset(dummy_data, labels=torch.ones(6)),
            transform=lambda x: x,
        )
        assert all(
            [
                dataset[torch.tensor(0)].data[m].shape == dummy_data[m].shape[1:]
                for m in dummy_data.keys()
            ]
        )
