"""Default architectures for the MultiVAE models."""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
from pythae.models.base.base_model import BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseEncoder
from torch import nn

from multivae.models.base.base_config import BaseAEConfig
from multivae.models.nn.base_architectures import (
    BaseConditionalDecoder,
    BaseJointEncoder,
    BaseMultilatentEncoder,
)


class Encoder_VAE_MLP(BaseEncoder):
    """Simple MLP encoder."""

    def __init__(self, args: dict, n_hidden=1):
        BaseEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))
        for _ in range(n_hidden):
            layers.append(nn.Sequential(nn.Linear(512, 512), nn.ReLU()))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(512, self.latent_dim)
        self.log_var = nn.Linear(512, self.latent_dim)

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

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i + 1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)

        return output


class Encoder_VAE_MLP_Style(BaseMultilatentEncoder):
    """A basic MLP encoders with two output embeddings.

    Args:
        args (BaseAEConfig): Contains input_dim, latent_dim, style_dim.

    Returns :
        ModelOutput(embedding = ..,
                    style_embedding = ..,
                    log_covariance = ..,
                    style_log_covariance = ..,)
    """

    def __init__(self, args: BaseAEConfig):
        BaseMultilatentEncoder.__init__(self)
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(np.prod(args.input_dim), 512), nn.ReLU()))

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

        out = x.reshape(-1, np.prod(self.input_dim))

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i + 1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out)
                output["log_covariance"] = self.log_var(out)
                output["style_embedding"] = self.style_embedding(out)
                output["style_log_covariance"] = self.style_log_var(out)

        return output


def BaseDictEncoders(input_dims: dict, latent_dim: int):
    """Given inputs dimensions and latent dimensions for all modalities,
        create a dictionary of basic MLP Encoders for each modality.

    Args:
        input_dims (dict): Containing the input dimension of each modality.
        latent_dim (int): Containing the desired latent dimension shared accross modalities.

    Returns:
        ~torch.nn.ModuleDict(): A Module Dictionary of Basic MLP Encoders.
    """
    encoders = nn.ModuleDict()
    for mod in input_dims:
        config = BaseAEConfig(input_dim=input_dims[mod], latent_dim=latent_dim)
        encoders[mod] = Encoder_VAE_MLP(config)
    return encoders


def BaseDictEncoders_MultiLatents(
    input_dims: dict, latent_dim: int, modality_dims: dict
):
    """The equivalent of BaseDictEncoders but with encoders that returns two latent variables : a content variable and a style variable.

    Args:
        input_dims (dict): Input dimensions of each modality
        latent_dim (int): Latent dimension for the shared latent space.
        modality_dims (dict): Latent dimensions for each modality specific latent space.

    Returns:
        nn.ModuleDict(): A Dictionary of basic MLP encoders.
    """
    encoders = nn.ModuleDict()
    for mod in input_dims:
        config = BaseAEConfig(
            input_dim=input_dims[mod],
            latent_dim=latent_dim,
            style_dim=modality_dims[mod],
        )
        encoders[mod] = Encoder_VAE_MLP_Style(config)
    return encoders


def BaseDictDecoders(input_dims: dict, latent_dim: int):
    """Given inputs dimensions and latent dimensions for all modalities,
        create a dictionary of basic MLP Decoders for each modality.

    Args:
        input_dims (dict): Containing the input dimension of each modality.
        latent_dim (int): latent dimension shared accross modalities.

    Returns:
        ~torch.nn.ModuleDict(): A Module Dictionary of Basic MLP Decoders.
    """
    decoders = nn.ModuleDict()
    for mod in input_dims:
        config = BaseAEConfig(input_dim=input_dims[mod], latent_dim=latent_dim)
        decoders[mod] = Decoder_AE_MLP(config)
    return decoders


def BaseDictDecodersMultiLatents(
    input_dims: dict, latent_dim: int, modality_dims: dict
):
    """The equivalent of BaseDictDecoders but for models with multiple latent spaces : a content variable and a style variable per modality.

    Args:
        input_dims (dict): Input dimensions of each modality
        latent_dim (int): Latent dimension for the shared latent space.
        modality_dims (dict): Latent dimensions for each modality specific latent space.

    Returns:
        nn.ModuleDict(): A Dictionary of basic MLP decoders.
    """
    decoders = nn.ModuleDict()
    for mod in input_dims:
        config = BaseAEConfig(
            input_dim=input_dims[mod], latent_dim=latent_dim + modality_dims[mod]
        )
        decoders[mod] = Decoder_AE_MLP(config)
    return decoders


class Decoder_AE_MLP(BaseDecoder):
    """The same as in Pythae but allows for any input shape (*, latent_dim)
    with * containing any number of dimensions.
    """

    def __init__(self, args: BaseAEConfig):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(args.latent_dim, 512), nn.ReLU()))

        layers.append(
            nn.Sequential(nn.Linear(512, int(np.prod(args.input_dim))), nn.Sigmoid())
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, **kwargs):
        output = ModelOutput()

        max_depth = self.depth
        out = z

        for i in range(max_depth):
            out = self.layers[i](out)
            if i + 1 == self.depth:
                output_shape = (*z.shape[:-1],) + self.input_dim
                output["reconstruction"] = out.reshape(output_shape)

        return output


class MultipleHeadJointEncoder(BaseJointEncoder):
    """A default instance of joint encoder created from copying the architectures for the unimodal encoders,
    concatenating their outputs and passing them through a unifying Multi-Layer-Perceptron.

    Args:
            dict_encoders (dict): Contains an instance of BaseEncoder for each modality (key).
            args (BaseAEConfig): config dictionary. Contains the latent dim.
            hidden_dim (int) : Default to 512.
            n_hidden_layers (int) : Default to 2.
    """

    def __init__(
        self,
        dict_encoders: dict,
        args: BaseAEConfig,
        hidden_dim=512,
        n_hidden_layers=2,
        **kwargs,
    ):
        super().__init__()

        # Duplicate all the unimodal encoders with identical instances.
        self.encoders = nn.ModuleDict()
        self.joint_input_dim = 0
        for modality in dict_encoders:
            self.encoders[modality] = deepcopy(dict_encoders[modality])
            self.joint_input_dim += self.encoders[modality].latent_dim

        modules = [
            nn.Sequential(nn.Linear(self.joint_input_dim, hidden_dim), nn.ReLU(True))
        ]
        for _ in range(n_hidden_layers - 1):
            modules.extend(
                [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))]
            )

        self.enc = nn.Sequential(*modules)
        self.fc1 = nn.Linear(hidden_dim, args.latent_dim)
        self.fc2 = nn.Linear(hidden_dim, args.latent_dim)

        self.latent_dim = args.latent_dim

    def forward(self, x: dict):
        """Implements the encoding of the data contained in x.

        Args:
            x (dict): Contains a tensor for each modality (key).
        """
        assert np.all(x.keys() == self.encoders.keys())

        modalities_outputs = []
        for mod in self.encoders:
            modalities_outputs.append(self.encoders[mod](x[mod])["embedding"])

        # Stack the modalities outputs
        concatened_outputs = torch.cat(modalities_outputs, dim=1)
        h = self.enc(concatened_outputs)
        embedding = self.fc1(h)
        log_covariance = self.fc2(h)
        output = ModelOutput(embedding=embedding, log_covariance=log_covariance)

        return output


class ConditionalDecoderMLP(BaseConditionalDecoder):
    """Base MLP Conditional Decoder for a single modality."""

    def __init__(
        self, latent_dim: int, cond_data_dims: Dict[str, tuple], data_dim: tuple
    ) -> ModelOutput:
        super().__init__()
        self.latent_dim = latent_dim

        self.all_dim = latent_dim
        for cond_data_dim in cond_data_dims.values():
            self.all_dim += np.prod(cond_data_dim)

        self.network = Decoder_AE_MLP(
            BaseAEConfig(input_dim=data_dim, latent_dim=self.all_dim)
        )

    def forward(self, z, cond_mods: Dict[str, torch.Tensor]):
        concatenated = torch.cat(
            [z] + [cond_data.view(z.shape[0], -1) for cond_data in cond_mods.values()],
            dim=1,
        )
        return self.network(concatenated)
