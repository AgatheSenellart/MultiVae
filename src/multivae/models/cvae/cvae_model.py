from typing import List, Union

import torch
import torch.distributions as dist

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base import BaseModel
from multivae.models.base.base_model import ModelOutput
from multivae.models.base.base_utils import set_decoder_dist
from multivae.models.nn.base_architectures import (
    BaseConditionalDecoder,
    BaseEncoder,
    BaseJointEncoder,
)
from multivae.models.nn.default_architectures import (
    BaseDictEncoders,
    ConditionalDecoder_MLP,
    MultipleHeadJointEncoder,
)

from .cvae_config import CVAEConfig


class CVAE(BaseModel):
    """Conditional Variational Autoencoder model.

    See https://arxiv.org/abs/1906.02691 for more information.
    """

    def __init__(
        self,
        model_config: CVAEConfig,
        encoder: Union[BaseEncoder, None] = None,
        decoder: Union[BaseConditionalDecoder, None] = None,
        prior_network: Union[BaseEncoder, None] = None,
    ):
        super().__init__(model_config)

        self.latent_dim = model_config.latent_dim
        self.beta = model_config.beta
        self.model_name = "CVAE"

        if model_config.decoder_dist_params is None:
            model_config.decoder_dist_params = {}

        self.set_decoder_dist(
            model_config.decoder_dist, model_config.decoder_dist_params
        )

        self.main_modality = model_config.main_modality
        self.conditioning_modality = model_config.conditioning_modality
        self.model_config = model_config

        self._set_encoder(encoder, model_config)
        self._set_decoder(decoder, model_config)
        self._set_prior_network(prior_network)

    def _set_encoder(self, encoder, model_config):
        if encoder is None:
            encoder = self.default_encoder(model_config)
        else:
            self.model_config.custom_architectures.append("encoder")

        if not isinstance(encoder, BaseJointEncoder):
            raise ValueError("The encoder must be an instance of BaseJointEncoder")

        self.encoder = encoder

    def _set_decoder(self, decoder, model_config):
        if decoder is None:
            decoder = self.default_decoder(model_config)

        else:
            self.model_config.custom_architectures.append("decoder")

        if not isinstance(decoder, BaseConditionalDecoder):
            raise ValueError(
                "The decoder must be an instance of BaseConditionalDecoder"
            )

        self.decoder = decoder

    def _set_prior_network(self, prior_network):
        if prior_network is None:
            self.prior_network = (
                None  # the prior will be fixed to a standard normal distribution
            )

        elif not isinstance(prior_network, BaseEncoder):
            raise ValueError("The prior network must be an instance of BaseEncoder")

        else:
            self.prior_network = prior_network
            self.model_config.custom_architectures.append("prior_network")

    def set_decoder_dist(self, dist, dist_params):
        self.recon_log_prob = set_decoder_dist(dist, dist_params)

    def default_encoder(self, model_config):
        if model_config.input_dims is None:
            raise AttributeError(
                "No encoder was provided but model_config.input_dims is None",
                "Please provide the input_dims of the model or an encoder architecture",
            )

        return MultipleHeadJointEncoder(
            dict_encoders=BaseDictEncoders(
                model_config.input_dims, model_config.latent_dim
            ),
            args=model_config,
            hidden_dim=512,
            n_hidden_layers=2,
        )

    def default_decoder(self, model_config):
        if model_config.input_dims is None:
            raise AttributeError(
                "No decoder was provided but model_config.input_dims is None",
                "Please provide the input_dims of the model or a decoder architecture",
            )

        return ConditionalDecoder_MLP(
            latent_dim=model_config.latent_dim,
            data_dim=model_config.input_dims[model_config.main_modality],
            conditioning_data_dim=model_config.input_dims[
                model_config.conditioning_modality
            ],
        )

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """
        Forward pass of the Conditional Variational Autoencoder.

        Args:
            inputs (dict): A dictionary containing the input data for each modality.

        Returns:
            ModelOutput : A ModelOutput instance containing the loss and metrics.
        """

        # Encode the input data
        output = self.encoder(inputs.data)
        embedding, log_var = output.embedding, output.log_covariance

        # Sample from the posterior
        z = dist.Normal(embedding, torch.exp(0.5 * log_var)).rsample()

        # Compute parameters of the prior p(z|conditioning_modality)

        if self.prior_network is None:
            prior_mean = torch.zeros_like(embedding)
            prior_log_var = torch.zeros_like(log_var)

        else:
            output = self.prior_network(inputs.data[self.conditioning_modality])
            prior_mean, prior_log_var = output.embedding, output.log_covariance

        # Compute the reconstruction loss of the main modality
        output = self.decoder(z, inputs.data[self.conditioning_modality])
        recon = output.reconstruction

        recon_loss = (
            -self.recon_log_prob(recon, inputs.data[self.main_modality]).mean(0).sum()
        )

        # Compute the KL divergence between the posterior and the prior
        kl_div = (
            self.kl_divergence(embedding, log_var, prior_mean, prior_log_var)
            .mean(0)
            .sum()
        )

        # Compute the total loss

        loss = recon_loss + kl_div * self.beta

        metrics = dict(kl=kl_div, recon_loss=recon_loss)

        return ModelOutput(loss=loss, metrics=metrics)

    def kl_divergence(self, mean, log_var, prior_mean, prior_log_var):
        kl = (
            0.5
            * (
                prior_log_var
                - log_var
                - 1
                + torch.exp(log_var - prior_log_var)
                + (mean - prior_mean) ** 2
            )
            / torch.exp(prior_log_var)
        )

        return kl.sum(dim=-1)

    def encode(
        self, inputs: MultimodalBaseDataset, N: int = 1, **kwargs
    ) -> ModelOutput:
        """Generate latent code by encoding the data and sampling from the
        posterior distribution.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.
            N (int, optional): number of samples per datapoint to sample from the posterior. Defaults to 1.

        Returns:
            A ModelOutput instance containing the embeddings. The shape of the embeddings is (N,batch_size,latent_dim)

            .. code-block:: python

                >>> output = model.encode(inputs, N=2)
                >>> z = output.z


        """
        return_mean = kwargs.pop("return_mean", False)
        flatten = kwargs.pop("flatten", False)

        output = self.encoder(inputs.data)
        mean, log_var = output.embedding, output.log_covariance
        scale = torch.exp(0.5 * log_var)
        sample_shape = [] if N == 1 else [N]

        if return_mean:
            z = torch.stack([mean] * N) if N > 1 else mean
        else:
            z = dist.Normal(mean, scale).rsample(sample_shape)

        if N > 1:
            cond_mod_data = torch.stack([inputs.data[self.conditioning_modality]] * N)
            if flatten:
                N, l, d = z.shape
                z = z.reshape(l * N, d)
                N, bs = cond_mod_data.shape[0], cond_mod_data.shape[1]
                cond_mod_data = cond_mod_data.reshape(N * bs, *cond_mod_data.shape[2:])
        else:
            cond_mod_data = inputs.data[self.conditioning_modality]

        return ModelOutput(z=z, cond_mod_data=cond_mod_data)

    def decode(self, embedding: ModelOutput, **kwargs) -> ModelOutput:
        """Decode embeddings to reconstruct the main modality.

        Returns:
            A ModelOutput instance containing the reconstruction.

            .. code-block:: python

                >>> embeddings = model.encode(inputs, N=2)
                >>> output = model.decode(embeddings)
                >>> output.reconstruction

        """
        z = embedding.z
        cond_mod_data = embedding.cond_mod_data

        if len(z.shape) == 3:
            N, l, d = z.shape
            z = z.reshape(l * N, d)
            cond_mod_data = cond_mod_data.reshape(N * l, *cond_mod_data.shape[2:])

            output = self.decoder(z, cond_mod_data)
            output.reconstruction = output.reconstruction.reshape(
                N, l, *output.reconstruction.shape[1:]
            )
            return output
        else:
            return self.decoder(z, cond_mod_data)

    def generate_from_prior(
        self, cond_mod_data: torch.Tensor, N: int = 1, **kwargs
    ) -> ModelOutput:
        """Generates latent variables from the prior, conditioning on cond_mod_data.

        Args :
            cond_mod_data (torch.Tensor) : Data from the conditioning modality.
            N (int) : number of latent codes to sample from the prior per datapoint


        Returns:
            A ModelOutput instance containing the embeddings.
        """

        flatten = kwargs.pop("flatten", False)

        if self.prior_network is None:
            prior_mean = torch.zeros((cond_mod_data.shape[0], self.latent_dim))
            prior_log_var = torch.zeros((cond_mod_data.shape[0], self.latent_dim))
        else:
            output = self.prior_network(cond_mod_data)
            prior_mean, prior_log_var = output.embedding, output.log_covariance

        sample_shape = [] if N == 1 else [N]
        z = dist.Normal(prior_mean, torch.exp(0.5 * prior_log_var)).rsample(
            sample_shape
        )

        if N > 1:
            cond_mod_data = torch.stack([cond_mod_data] * N)
            if flatten:
                N, l, d = z.shape
                z = z.reshape(l * N, d)
                N, bs = cond_mod_data.shape[0], cond_mod_data.shape[1]
                cond_mod_data = cond_mod_data.reshape(N * bs, *cond_mod_data.shape[2:])

        return ModelOutput(z=z, cond_mod_data=cond_mod_data)

    def predict(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N=1,
        **kwargs,
    ) -> ModelOutput:
        """Reconstruct from the input or from the conditioning modality.


        Args:
            inputs (MultimodalBaseDataset) : The data to use for prediction.
            cond_mod (list or str) : Either 'all' to perform reconstruction or the name of the conditioning modality to
                sample from the prior using the conditioning modality.
        Returns:
            ModelOutput : A ModelOutput instance containing the reconstruction / generation.

            .. code-block:: python

                >>> # reconstructions
                >>> output = model.predict(inputs, cond_mod = 'all')
                >>> reconstruction = output.reconstruction




        """

        if type(cond_mod) == str:
            if cond_mod == "all":
                cond_mod = [self.main_modality, self.conditioning_modality]
            else:
                cond_mod = [cond_mod]

        if len(cond_mod) == 1:
            if cond_mod[0] == self.conditioning_modality:
                embeddings = self.generate_from_prior(
                    cond_mod_data=inputs.data[self.conditioning_modality], N=N, **kwargs
                )
            elif cond_mod[0] == self.main_modality:
                embeddings = self.encode(inputs, N, **kwargs)
            else:
                raise ValueError(
                    "The conditioning modality must be one of the modalities of the model"
                    ". You provided {}".format(cond_mod[0])
                )
        elif len(cond_mod) == 2:
            if (
                self.conditioning_modality not in cond_mod
                or self.main_modality not in cond_mod
            ):
                raise ValueError(
                    "One of the modalities in cond_mod is not part of the model"
                    "cond_mod : {}".format(cond_mod)
                )
            embeddings = self.encode(inputs, N, **kwargs)

        else:
            raise ValueError(
                "The length of cond_mod must be 1 or 2 since the model is a CVAE"
            )

        output = self.decode(embeddings)
        return output
