""" Custom encoders and decoders for testing purposes.
"""
from typing import List

import numpy as np
import torch
from pythae.models.base.base_model import BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseEncoder
from torch import nn


class EncoderTest(BaseEncoder):
    """A simple MLP, but different than the default ones."""

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 64), nn.ReLU()))

        for i in range(1):
            layers.append(nn.Sequential(nn.Linear(64, 64), nn.ReLU()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(64, self.latent_dim)
        self.log_var = nn.Linear(64, self.latent_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(
            -1,
            np.prod(self.input_dim),
        )

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)

        return output


class EncoderTestMultilatents(BaseEncoder):
    """Encoder for models that use multiple latent spaces."""

    def __init__(self, args: dict):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim

        layers = nn.ModuleList()
        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))

        for i in range(1):
            layers.append(nn.Sequential(nn.Linear(512, 512), nn.ReLU()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)
        self.style_embedding = nn.Linear(512, self.style_dim)
        self.style_log_var = nn.Linear(512, self.style_dim)

    def forward(self, x, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x.reshape(
            -1,
            np.prod(self.input_dim),
        )

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)
                output["style_embedding"] = self.style_embedding(out)
                output["style_log_covariance"] = self.style_log_var(out)

        return output


class DecoderTest(BaseDecoder):
    """A simple MLP decoder, different than the default one."""
    def __init__(self, args: dict):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 64), nn.ReLU()))

        layers.append(
            nn.Sequential(nn.Linear(64, int(np.prod(args.input_dim))), nn.Sigmoid())
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:
            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth}). "
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["reconstruction"] = out.reshape((z.shape[0],) + self.input_dim)

        return output
