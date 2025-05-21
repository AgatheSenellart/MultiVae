import logging
from typing import Dict, Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base import BaseDecoder, BaseEncoder
from multivae.models.nn.default_architectures import (
    BaseAEConfig,
    Decoder_AE_MLP,
    Encoder_VAE_MLP,
    nn,
)

from ..base import BaseMultiVAE
from ..base.base_utils import rsample_from_gaussian
from .nexus_config import NexusConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class Nexus(BaseMultiVAE):
    """The Nexus model from (Vasco et al 2022)
    "Leveraging hierarchy in multimodal generative models for effective cross-modality inference".


    Args:
        model_config (NexusConfig): An instance of NexusConfig in which any model's parameters is
            made available.

        encoders (Dict[str, ~pythae.models.nn.BaseEncoder]): A dictionary
            containing the modalities names and the encoders for each modality. Each encoder is
            an instance of ~pythae.models.nn.BaseEncoder

        decoders (Dict[str, ~pythae.models.nn.BaseDecoder]): A dictionary
            containing the modalities names and the decoders for each modality. Each decoder is an
            instance of ~pythae.models.nn.BaseDecoder

        top_encoders (Dict[str, ~pythae.models.nn.BaseEncoder]) : A dictionary containing for each modality,
            the top encoder to use.

        joint_encoder (~multivae.models.nn.BaseJointEncoder): The encoder that takes the aggregated message and
            encode it to obtain the high level latent distribution.

        top_decoders (Dict[str, ~pythae.models.nn.BaseDecoder]) : A dictionary containing for each modality, the top decoder to use.

    """

    def __init__(
        self,
        model_config: NexusConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        top_encoders: Dict[str, BaseEncoder] = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        top_decoders: Dict[str, BaseEncoder] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, **kwargs)
        self.model_name = "NEXUS"

        # Set all architectures
        self._set_top_decoders(top_decoders, model_config)
        self._set_top_encoders(top_encoders, model_config)
        self._set_joint_encoder(joint_encoder, model_config)

        self._set_bottom_betas(model_config.bottom_betas)
        self._set_gammas(model_config.gammas)

        self.start_keep_best_epoch = model_config.warmup + 1  # important for training.
        self.adapt_top_decoder_variance = self._set_top_decoder_variance(model_config)
        self.check_aggregator(model_config)

    def _compute_bottom_elbos(self, inputs: MultimodalBaseDataset, **kwargs):
        """Passes the inputs through the first level of encoding and compute the bottom elbos."""
        epoch = kwargs.pop("epoch", 1)
        annealing = min(epoch / self.model_config.warmup, 1.0)

        # Compute the first level representations and ELBOs
        modalities_msg = {}
        bottom_loss = 0
        first_level_z = {}
        metrics = {}

        for m, x_m in inputs.data.items():
            # Encode the modality
            output_m = self.encoders[m](x_m)
            z_m = rsample_from_gaussian(output_m.embedding, output_m.log_covariance)

            # Decode and reconstruct
            recon_x_m = self.decoders[m](z_m).reconstruction

            # Compute -log p(x_m|z_m)
            nlogprob = (
                -(self.recon_log_probs[m](recon_x_m, x_m) * self.rescale_factors[m])
                .reshape(recon_x_m.size(0), -1)
                .sum(-1)
            )

            # Compute KL(q(z_m|x_m)||p(z_m)). p(z_m) is a standard gaussian
            KLD = -0.5 * torch.sum(
                1
                + output_m.log_covariance
                - output_m.embedding.pow(2)
                - output_m.log_covariance.exp(),
                dim=-1,
            )

            # Compute the negative elbo loss
            m_elbo = nlogprob + KLD * self.bottom_betas[m] * annealing

            # Save a detached z
            first_level_z[m] = z_m.clone().detach()
            # Pass the modality specific latent variable through the top encoder to compute the message
            modalities_msg[m] = self.top_encoders[m](first_level_z[m]).embedding
            # Save some metrics for monitoring the training.
            metrics["recon_loss_" + m] = nlogprob.mean()
            metrics["kl_" + m] = KLD.mean()

            # Partial dataset : use masks to filter out unavailable samples in the loss
            if hasattr(inputs, "masks"):
                m_elbo = m_elbo * inputs.masks[m].float()

            bottom_loss += m_elbo

        return bottom_loss, modalities_msg, first_level_z, metrics

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """Forward pass of the model. Returns loss and metrics."""
        # Compute bottom level elbos
        bottom_loss, modalities_msg, first_level_z, metrics = (
            self._compute_bottom_elbos(inputs, **kwargs)
        )

        # Aggregate the modalities messages
        aggregated_msg = self._aggregate_during_training(inputs, modalities_msg)

        # Compute the higher level latent variable z_\sigma
        joint_output = self.joint_encoder(aggregated_msg)
        joint_z = rsample_from_gaussian(
            joint_output.embedding, joint_output.log_covariance
        )

        # Compute log p(z_m|z_sigma)
        z_recon_loss = 0
        for m in self.top_decoders:
            z_m_recon = self.top_decoders[m](joint_z).reconstruction

            # Eventually adapt the scale of the top decoder
            if m in self.adapt_top_decoder_variance:
                scale = (
                    ((first_level_z[m] - z_m_recon) ** 2)
                    .mean([0, 1], keepdim=True)
                    .sqrt()
                )
            else:
                scale = 1

            z_m_recon_loss = (
                -(dist.Normal(z_m_recon, scale).log_prob(first_level_z[m])).sum(-1)
                * self.gammas[m]
            )

            # Partial dataset, we don't reconstruct the missing modalities
            if hasattr(inputs, "masks"):
                z_m_recon_loss = z_m_recon_loss * inputs.masks[m]

            z_recon_loss += z_m_recon_loss
            metrics["recon_z_" + m] = (
                z_m_recon_loss.mean()
            )  # save metrics for monitoring

        # Compute KL(q(z_sigma|z1::M) | p(z_sigma)). The prior p(z_sigma) is standard gaussian
        joint_KLD = -0.5 * torch.sum(
            1
            + joint_output.log_covariance
            - joint_output.embedding.pow(2)
            - joint_output.log_covariance.exp(),
            dim=1,
        )
        epoch = kwargs.pop("epoch", 1)
        annealing = min(epoch / self.model_config.warmup, 1.0)

        # Compute top loss and total loss
        top_loss = z_recon_loss + self.model_config.top_beta * joint_KLD * annealing
        total_loss = top_loss + bottom_loss

        metrics.update(
            {
                "annealing": annealing,
                "bottom_loss": bottom_loss.mean(0),
                "top_loss": top_loss.mean(0),
                "joint_KLD": joint_KLD.mean(0),
            }
        )
        # Return the mean averaged on the batch.
        return ModelOutput(
            loss=total_loss.mean(0),
            loss_sum=total_loss.sum(),
            metrics=metrics,
        )

    def _aggregate_during_training(
        self, inputs: MultimodalBaseDataset, modalities_msg: dict
    ):
        """Aggregate the modalities during training. It applies the forced perceptual dropout if the dataset is not already incomplete."""
        if self.model_config.aggregator == "mean":
            # With an already incomplete dataset, we don't apply dropout
            if hasattr(inputs, "masks"):
                normalization_per_sample = torch.stack(
                    [inputs.masks[m] for m in inputs.masks], dim=0
                ).sum(0)
                # Apply the masks and sum
                aggregated_msg = 0
                for m, msg in modalities_msg.items():
                    aggregated_msg += msg * inputs.masks[m].unsqueeze(1)
                # Normalize
                aggregated_msg = (aggregated_msg.t() / normalization_per_sample).t()

            # With a complete dataset, we apply Forced Perceptual Dropout during training
            else:
                # before stack , we have n_modalities tensor of shape n_data, msg_dim.
                # After we have one single tensor of shape n_data, n_modalities, msg_dim
                tensor_modalities_msg = torch.stack(
                    list(modalities_msg.values()), dim=1
                )
                batch_msgs = []

                # we iter over the batch samples
                for msgs in tensor_modalities_msg:
                    # msgs shape : n_modalities, msg_dim
                    bernoulli_drop = (
                        dist.Bernoulli(self.model_config.dropout_rate).sample().item()
                    )

                    if bernoulli_drop == 1:
                        # choose a random subset to keep
                        subset_size = np.random.randint(1, self.n_modalities)
                        msgs = msgs[torch.randperm(self.n_modalities)]
                        msgs = msgs[:subset_size]

                    batch_msgs.append(msgs.mean(0))

                aggregated_msg = torch.stack(batch_msgs, dim=0)

            return aggregated_msg

        return

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        return_mean=False,
        **kwargs,
    ):
        """Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities
                names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.
            return_mean (bool) : if True, returns the mean of the posterior distribution (instead of a sample).


        Returns:
            ModelOutput instance with fields:
                z (torch.Tensor (n_data, N, latent_dim))
                one_latent_space (bool) = True

        """
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod
        modalities_z = {}
        modalities_msg = {}
        flatten = kwargs.pop("flatten", False)

        # Encode each modality with the bottom encoders
        for m in cond_mod:
            output_m = self.encoders[m](inputs.data[m])
            modalities_z[m] = rsample_from_gaussian(
                output_m.embedding,
                output_m.log_covariance,
                N,
                return_mean,
                flatten=True,
            )
            modalities_msg[m] = self.top_encoders[m](modalities_z[m]).embedding

        # Compute aggregated msg
        aggregated_msg = None
        if self.model_config.aggregator == "mean":
            aggregated_msg = torch.stack(list(modalities_msg.values()), dim=0).mean(0)
        nexus_output = self.joint_encoder(aggregated_msg)
        z = rsample_from_gaussian(
            nexus_output.embedding,
            nexus_output.log_covariance,
            N=1,
            return_mean=return_mean,
        )

        if N > 1 and not flatten:
            z = z.reshape(N, -1, *z.shape[1:])
            modalities_z = {
                m: modalities_z[m].reshape(N, -1, *modalities_z[m].shape[1:])
                for m in modalities_z
            }

        return ModelOutput(z=z, one_latent_space=True, modalities_z=modalities_z)

    def decode(
        self, embedding: ModelOutput, modalities: Union[list, str] = "all", **kwargs
    ):
        """Decodes the embeddings given by the latent function."""
        self.eval()
        with torch.no_grad():
            if modalities == "all":
                modalities = list(self.encoders.keys())
            elif isinstance(modalities, str):
                modalities = [modalities]

            # For self reconstruction, we use the bottom encodings.
            use_bottom_z_for_reconstruction = kwargs.pop("use_bottom_z_for_recon", True)
            if not hasattr(embedding, "modalities_z"):
                use_bottom_z_for_reconstruction = False

            outputs = ModelOutput()

            # If the embedding has three dimensions, we flatten it and then reshape it at the end.
            reshape = False
            if len(embedding.z.shape) == 3:
                N, bs, _ = embedding.z.shape
                reshape = True

            for m in modalities:
                if (use_bottom_z_for_reconstruction) and (
                    m in embedding.modalities_z.keys()
                ):
                    z_m = embedding.modalities_z[m]
                    if reshape:
                        z_m = z_m.view(N * bs, -1)
                else:
                    z = embedding.z
                    if reshape:
                        z = z.view(N * bs, -1)
                    z_m = self.top_decoders[m](z).reconstruction

                recon = self.decoders[m](z_m).reconstruction
                if reshape:
                    recon = recon.reshape(N, bs, *recon.shape[1:])
                outputs[m] = recon

            return outputs

    def _set_top_decoder_variance(self, config):
        """Returns a list of the modalities for which the variance needs to be adapted."""
        if config.adapt_top_decoder_variance is None:
            return []

        for m in config.adapt_top_decoder_variance:
            if m not in self.modalities_name:
                raise AttributeError(
                    f"A string provided in *adapt_top_decoder_variance* field doesn't match any of the modalities name : {m} is not in {self.modalities_name}"
                )
        return config.adapt_top_decoder_variance

    def _set_bottom_betas(self, bottom_betas):
        if bottom_betas is None:
            bottom_betas = {m: 1.0 for m in self.encoders}

        if bottom_betas.keys() != self.encoders.keys():
            raise AttributeError(
                "The bottom_betas keys do not match the modalitiesnames in encoders."
            )

        self.bottom_betas = bottom_betas

    def _set_gammas(self, gammas):
        if gammas is None:
            self.gammas = {m: 1.0 for m in self.encoders}
        elif gammas.keys() != self.encoders.keys():
            raise AttributeError(
                "The gammas keys do not match the modalitiesnames in encoders."
            )
        else:
            self.gammas = gammas

    def default_encoders(self, model_config: NexusConfig):
        if (
            model_config.input_dims is None
            or model_config.modalities_specific_dim is None
        ):
            raise AttributeError(
                "Please provide encoders architectures or "
                "valid input_dims and modalities_specific_dim in the"
                "model configuration"
            )

        encoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(
                input_dim=model_config.input_dims[mod],
                latent_dim=model_config.modalities_specific_dim[mod],
            )
            encoders[mod] = Encoder_VAE_MLP(config)
        return encoders

    def default_decoders(self, model_config: NexusConfig):
        if (
            model_config.input_dims is None
            or model_config.modalities_specific_dim is None
        ):
            raise AttributeError(
                "Please provide decoders architectures or "
                "valid input_dims and modalities_specific_dim in the"
                "model configuration"
            )

        decoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(
                input_dim=model_config.input_dims[mod],
                latent_dim=model_config.modalities_specific_dim[mod],
            )
            decoders[mod] = Decoder_AE_MLP(config)
        return decoders

    def _default_top_encoders(self, model_config: NexusConfig):
        if model_config.modalities_specific_dim is None:
            raise AttributeError(
                "Please provide top_encoders architectures or "
                "valid modalities_specific_dim in the"
                "model configuration"
            )

        encoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(
                input_dim=(model_config.modalities_specific_dim[mod],),
                latent_dim=model_config.msg_dim,
            )
            encoders[mod] = Encoder_VAE_MLP(config)
        return encoders

    def _default_top_decoders(self, model_config: NexusConfig):
        if model_config.modalities_specific_dim is None:
            raise AttributeError(
                "Please provide top_decoders architectures or "
                "valid modalities_specific_dim in the"
                "model configuration"
            )

        decoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(
                input_dim=(model_config.modalities_specific_dim[mod],),
                latent_dim=model_config.latent_dim,
            )
            decoders[mod] = Decoder_AE_MLP(config)
        return decoders

    def _default_joint_encoder(self, model_config: NexusConfig):
        return Encoder_VAE_MLP(
            BaseAEConfig(
                input_dim=(model_config.msg_dim,), latent_dim=model_config.latent_dim
            )
        )

    def _set_top_encoders(self, top_encoders, model_config):
        # Provide default encoders if None are provided
        if top_encoders is None:
            top_encoders = self._default_top_encoders(model_config)
        else:
            self.model_config.custom_architectures.append("top_encoders")
        # Check top encoders type and set the attribute
        self.top_encoders = nn.ModuleDict()
        for k in top_encoders:
            if not isinstance(top_encoders[k], BaseEncoder):
                raise AttributeError(
                    "Top Encoders must be instances of multivae.models.base.BaseEncoder"
                )
            self.top_encoders[k] = top_encoders[k]

    def _set_top_decoders(self, top_decoders, model_config):
        # Provide default MLP decoders if None are provided.
        if top_decoders is None:
            top_decoders = self._default_top_decoders(model_config)
        else:
            self.model_config.custom_architectures.append("top_decoders")

        # Check the decoders type and set the attribute
        self.top_decoders = nn.ModuleDict()
        for k in top_decoders:
            if not isinstance(top_decoders[k], BaseDecoder):
                raise AttributeError(
                    "Top Decoders must be instances of multivae.models.base.BaseDecoder"
                )
            self.top_decoders[k] = top_decoders[k]

    def _set_joint_encoder(self, joint_encoder, model_config):
        # Provide default Nexus encoder if None is Provided
        if joint_encoder is None:
            joint_encoder = self._default_joint_encoder(model_config)
        else:
            self.model_config.custom_architectures.append("joint_encoder")

        # Check encoder type and set the attribute
        if not isinstance(joint_encoder, BaseEncoder):
            raise AttributeError(
                "Joint encoder must be an instance of multivae.models.base.BaseEncoder"
            )
        self.joint_encoder = joint_encoder

    def check_aggregator(self, model_config):
        if model_config.aggregator not in ["mean"]:
            raise AttributeError(
                f"This aggregator {model_config.aggregator} is not supported at the moment"
            )
