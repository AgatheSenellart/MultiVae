from typing import Dict, Union

import torch
import torch.distributions as dist

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base import BaseModel, ModelOutput
from multivae.models.base.base_utils import kl_divergence, set_decoder_dist
from multivae.models.nn.base_architectures import (
    BaseConditionalDecoder,
    BaseEncoder,
    BaseJointEncoder,
)
from multivae.models.nn.default_architectures import (
    BaseDictEncoders,
    ConditionalDecoderMLP,
    MultipleHeadJointEncoder,
)

from .cvae_config import CVAEConfig


class CVAE(BaseModel):
    """Main class for the Conditional Variational Autoencoder.

    Args:
        model_config (CVAEConfig): the model configuration class.
        encoder (BaseEncoder): The encoder network.
        decoder (BaseConditionalDecoder): The conditional decoder network.
        prior_network (BaseJointEncoder): Takes the conditional modalities and returns the
            parameters for the prior distribution.
    """

    def __init__(
        self,
        model_config: CVAEConfig,
        encoder: Union[BaseEncoder, None] = None,
        decoder: Union[BaseConditionalDecoder, None] = None,
        prior_network: Union[BaseJointEncoder, None] = None,
    ):
        super().__init__(model_config)

        self.latent_dim = model_config.latent_dim
        self.model_name = "CVAE"

        if model_config.decoder_dist_params is None:
            model_config.decoder_dist_params = {}

        self._set_decoder_dist(
            model_config.decoder_dist, model_config.decoder_dist_params
        )

        self.main_modality = model_config.main_modality
        self.conditioning_modalities = model_config.conditioning_modalities
        self.model_config = model_config

        self._set_encoder(encoder, model_config)
        self._set_decoder(decoder, model_config)
        self._set_prior_network(prior_network)

    def _set_encoder(self, encoder, model_config):
        if encoder is None:
            encoder = self._default_encoder(model_config)
        else:
            self.model_config.custom_architectures.append("encoder")

        if not isinstance(encoder, BaseJointEncoder):
            raise ValueError("The encoder must be an instance of BaseJointEncoder")

        self.encoder = encoder

    def _set_decoder(self, decoder, model_config):
        if decoder is None:
            decoder = self._default_decoder(model_config)

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

        elif not isinstance(prior_network, BaseJointEncoder):
            raise ValueError(
                "The prior network must be an instance of BaseJointEncoder"
            )

        else:
            self.prior_network = prior_network
            self.model_config.custom_architectures.append("prior_network")

    def _set_decoder_dist(self, dist_name: str, dist_params):
        self.recon_log_prob = set_decoder_dist(dist_name, dist_params)

    def _default_encoder(self, model_config):
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

    def _default_decoder(self, model_config):
        if model_config.input_dims is None:
            raise AttributeError(
                "No decoder was provided but model_config.input_dims is None",
                "Please provide the input_dims of the model or a decoder architecture",
            )

        return ConditionalDecoderMLP(
            latent_dim=model_config.latent_dim,
            data_dim=model_config.input_dims[model_config.main_modality],
            cond_data_dims={
                mod: model_config.input_dims[mod]
                for mod in model_config.conditioning_modalities
            },
        )

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Forward pass of the Conditional Variational Autoencoder.

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
        cond_mod_data = {mod: inputs.data[mod] for mod in self.conditioning_modalities}

        if self.prior_network is None:
            prior_mean = torch.zeros_like(embedding)
            prior_log_var = torch.zeros_like(log_var)

        else:
            output = self.prior_network(cond_mod_data)
            prior_mean, prior_log_var = output.embedding, output.log_covariance

        # Compute the reconstruction loss of the main modality
        output = self.decoder(z, cond_mod_data)
        recon = output.reconstruction

        recon_loss = (
            -self.recon_log_prob(recon, inputs.data[self.main_modality]).mean(0).sum()
        )

        # Compute the KL divergence between the posterior and the prior
        kl_div = kl_divergence(embedding, log_var, prior_mean, prior_log_var).mean(0)

        # Compute the total loss

        loss = recon_loss + kl_div * self.model_config.beta

        metrics = {"kl": kl_div, "recon_loss": recon_loss}

        return ModelOutput(loss=loss, metrics=metrics)

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

        if N > 1 and not flatten:
            cond_mod_data = {
                m: torch.stack([inputs.data[m]] * N)
                for m in self.conditioning_modalities
            }

        elif N > 1 and flatten:
            cond_mod_data = {
                m: torch.cat([inputs.data[m]] * N) for m in self.conditioning_modalities
            }
            z = z.reshape(N * mean.shape[0], mean.shape[1])
        else:
            cond_mod_data = {m: inputs.data[m] for m in self.conditioning_modalities}

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
        with torch.no_grad():
            z = embedding.z
            cond_mod_data = embedding.cond_mod_data

            if len(z.shape) == 3:
                N, l, d = z.shape
                z = z.reshape(l * N, d)
                cond_mod_data = {
                    m: cond_mod_data[m].reshape(N * l, *cond_mod_data[m].shape[2:])
                    for m in cond_mod_data
                }

                output = self.decoder(z, cond_mod_data)
                output.reconstruction = output.reconstruction.reshape(
                    N, l, *output.reconstruction.shape[1:]
                )
                return output
            else:
                return self.decoder(z, cond_mod_data)

    def generate_from_prior(
        self, cond_mod_data: Dict[str, torch.Tensor], N: int = 1, **kwargs
    ) -> ModelOutput:
        """Generates latent variables from the prior, conditioning on cond_mod_data.

        Args :
            cond_mod_data (Dict[str,torch.Tensor]) : Data from the conditioning modality.
            N (int) : number of latent codes to sample from the prior per datapoint


        Returns:
            A ModelOutput instance containing the embeddings.
        """
        flatten = kwargs.pop("flatten", False)

        # Look up the batch size and the device of the input data
        batch_size = list(cond_mod_data.values())[0].shape[0]
        device = list(cond_mod_data.values())[0].device

        if self.prior_network is None:
            prior_mean = torch.zeros((batch_size, self.latent_dim))
            prior_log_var = torch.zeros((batch_size, self.latent_dim))
        else:
            output = self.prior_network(cond_mod_data)
            prior_mean, prior_log_var = output.embedding, output.log_covariance

        sample_shape = [] if N == 1 else [N]
        z = dist.Normal(prior_mean, torch.exp(0.5 * prior_log_var)).rsample(
            sample_shape
        )

        if N > 1 and not flatten:
            cond_mod_data = {
                m: torch.stack([cond_mod_data[m]] * N)
                for m in self.conditioning_modalities
            }

        elif N > 1 and flatten:
            cond_mod_data = {
                m: torch.cat([cond_mod_data[m]] * N)
                for m in self.conditioning_modalities
            }
            z = z.reshape(N * prior_mean.shape[0], prior_mean.shape[1])
        else:
            cond_mod_data = {m: cond_mod_data[m] for m in self.conditioning_modalities}

        z = z.to(device)

        return ModelOutput(z=z, cond_mod_data=cond_mod_data)

    def predict(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[str, list] = "all",
        N=1,
        **kwargs,
    ) -> ModelOutput:
        """Reconstruct from the input or from the conditioning modalities.

        Args:
            inputs (MultimodalBaseDataset) : The data to use for prediction.
            cond_mod (Union[str, list]) : Either 'all' to perform reconstruction or the list of conditioning modalities to generate from the prior.
            N (int) : number of samples per datapoint to sample from the posterior or prior.

        Returns:
            ModelOutput : A ModelOutput instance containing the reconstruction / generation.

            .. code-block:: python

                >>> # reconstructions
                >>> output = model.predict(inputs, cond_mod = 'all')
                >>> reconstruction = output.reconstruction




        """
        if (
            cond_mod == "all"
            or set(cond_mod) == set([self.main_modality])
            or set(cond_mod) == set([self.main_modality] + self.conditioning_modalities)
        ):
            embeddings = self.encode(inputs, N, **kwargs)

        elif set(cond_mod) == set(self.conditioning_modalities):
            cond_mod_data = {m: inputs.data[m] for m in self.conditioning_modalities}
            embeddings = self.generate_from_prior(cond_mod_data, N, **kwargs)

        else:
            raise ValueError(
                "The conditioning modalities must be either 'all' or the list of conditioning modalities"
            )

        output_decoder = self.decode(embeddings)

        output = ModelOutput()
        output[self.main_modality] = output_decoder.reconstruction

        return output
