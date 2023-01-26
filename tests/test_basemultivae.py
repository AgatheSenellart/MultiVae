import os

import pytest

from multivae.models import BaseMultiVAEConfig


class Test_BaseMultiVAEConfig:
    @pytest.fixture(
        params=[
            {"input_dim": (1, 3, 8), "latent_dim": 10},
            {"input_dim": None, "latent_dim": 15},
        ]
    )
    def input_latent_pairs(self, request):
        return request.param

    def test_create_config(self, input_latent_pairs):
        config = BaseMultiVAEConfig(**input_latent_pairs)

        assert config.input_dim == input_latent_pairs["input_dim"]
        assert config.latent_dim == input_latent_pairs["latent_dim"]
