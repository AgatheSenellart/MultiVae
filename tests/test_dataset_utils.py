import os

import numpy as np
import pytest
import torch

from multivae.data.datasets import DatasetOutput, MultimodalBaseDataset
from multivae.data.datasets.utils import ResampleDataset, adapt_shape
from multivae.data.utils import drop_unused_modalities, get_batch_size


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


@pytest.fixture
def dummy_masks():
    return dict(
        mod1=torch.bernoulli(torch.ones(6) * 1.0).bool(),
        mod2=torch.bernoulli(torch.ones(6) * 0.5).bool(),
        mod3=torch.bernoulli(torch.ones(6) * 0.0).bool(),
        mod4=torch.bernoulli(torch.ones(6) * 0.5).bool(),
        mod5=torch.tensor([0] * 5 + [1]).bool(),
        mod6=torch.bernoulli(torch.ones(6) * 0.0).bool(),
    )


@pytest.fixture
def dummy_dataset(dummy_data, dummy_masks):
    return DatasetOutput(data=dummy_data, masks=dummy_masks)


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


class Test_get_batch_size:
    def test(self, dummy_dataset):
        l = get_batch_size(dummy_dataset)
        assert l == 6


class Test_drop_modalities:
    def test(self, dummy_dataset):
        dummy_dataset = drop_unused_modalities(dummy_dataset)

        assert not "mod3" in dummy_dataset.data.keys()
        assert not "mod3" in dummy_dataset.masks.keys()

        assert not "mod6" in dummy_dataset.data.keys()
        assert not "mod6" in dummy_dataset.masks.keys()

        assert "mod1" in dummy_dataset.data.keys()
        assert "mod1" in dummy_dataset.masks.keys()

        assert "mod5" in dummy_dataset.data.keys()
        assert "mod5" in dummy_dataset.masks.keys()
