import os

import pytest

from multivae.models import BaseMultiVAEConfig


class Test_BaseMultiVAEConfig:
    @pytest.fixture(
        params=[
            dict(
                n_modalities=3,
                latent_dim=10,
                decoders_dist=dict(mod1="laplace", mod2="laplace"),
                decoder_dist_params=dict(mod1={"scale": 0.75}, mod2={"scale": 0.75}),
            ),
            dict(
                n_modalities=2,
                latent_dim=5,
                input_dims=dict(mod1=(2,), mod2=(3,)),
                decoders_dist=dict(mod1="laplace", mod2="laplace"),
                decoder_dist_params=dict(mod1={"scale": 0.75}, mod2={"scale": 0.75}),
            ),
        ]
    )
    def input_latent_pairs(self, request):
        return request.param

    def test_create_config(self, input_latent_pairs):
        config = BaseMultiVAEConfig(**input_latent_pairs)

        assert config.n_modalities == input_latent_pairs["n_modalities"]
        assert config.latent_dim == input_latent_pairs["latent_dim"]
        assert config.decoders_dist == input_latent_pairs["decoders_dist"]
        assert config.decoder_dist_params == input_latent_pairs["decoder_dist_params"]
