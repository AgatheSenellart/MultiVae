import os

import pytest

from multivae.models import BaseMultiVAEConfig


class Test_BaseMultiVAEConfig:
    @pytest.fixture(
        params=[
            {"n_modalities": 3, "latent_dim": 10},
            {"n_modalities": None, "latent_dim": 15},
        ]
    )
    def input_latent_pairs(self, request):
        return request.param

    def test_create_config(self, input_latent_pairs):
        config = BaseMultiVAEConfig(**input_latent_pairs)

        assert config.n_modalities == input_latent_pairs["n_modalities"]
        assert config.latent_dim == input_latent_pairs["latent_dim"]
