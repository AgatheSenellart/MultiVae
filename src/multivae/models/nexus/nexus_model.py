import logging
from typing import Dict, Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.normalizing_flows.base import BaseNF
from pythae.models.normalizing_flows.maf import MAF, MAFConfig
from torch.nn import ModuleDict

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base import BaseDecoder, BaseEncoder
from multivae.models.nn.default_architectures import (
    BaseAEConfig,
    Decoder_AE_MLP,
    Encoder_VAE_MLP,
    nn,
)

from ...data.datasets.base import MultimodalBaseDataset
from ..base import BaseMultiVAE
from .nexus_config import NexusConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class Nexus(BaseMultiVAE):

    """
    The Nexus model from
     "Leveraging hierarchy in multimodal generative models for effective cross-modality inference" (Vasco et al 2022)


    Args:

        model_config (NexusConfig): An instance of NexusConfig in which any model's parameters is
            made available.

        encoders (Dict[str, ~multivae.models.base.BaseEncoder]): A dictionary
            containing the modalities names and the encoders for each modality. Each encoder is
            an instance of Multivae's BaseEncoder whose output is of the form:
                ```ModelOutput(
                    embedding = ...,
                    log_covariance = ...,
                )

        decoders (Dict[str, ~multivae.models.base.BaseDecoder]): A dictionary
            containing the modalities names and the decoders for each modality. Each decoder is an
            instance of Pythae's BaseDecoder.

        top_encoders (Dict[str, ~multivae.models.base.BaseEncoder]) : An instance of
            BaseEncoder that takes all the first level representations to generate the messages that will be aggregated.
            Each encoder is
            an instance of Multivae's BaseEncoder whose output is of the form:
                ```ModelOutput(
                    embedding = ...,
                    log_covariance = ...,
                )

        joint_encoder (~multivae.models.base.BaseEncoder): The encoder that takes the aggregated message and
            encode it to obtain the high level latent distribution.

        top_decoders (Dict[str, ~multivae.models.base.BaseDecoder]) : Top level decoders from the joint representation
            to the modalities specific representations.

    """

    def __init__(
        self,
        model_config: NexusConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        top_encoders: Dict[str, BaseEncoder] = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        top_decoders: Dict[str, BaseNF] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, **kwargs)

        if top_encoders is None:
            top_encoders = self.default_top_encoders(model_config)
        else:
            self.model_config.custom_architectures.append("top_encoders")

        if top_decoders is None:
            top_decoders = self.default_top_decoders(model_config)
        else:
            self.model_config.custom_architectures.append("top_decoders")

        if joint_encoder is None:
            joint_encoder = self.default_joint_encoder(model_config)
        else:
            self.model_config.custom_architectures.append("joint_encoder")

        self.set_top_decoders(top_decoders)
        self.set_top_encoders(top_encoders)
        self.set_joint_encoder(joint_encoder)

        self.model_name = "NEXUS"

        self.dropout = model_config.dropout_rate
        self.set_bottom_betas(model_config.bottom_betas)
        self.set_gammas(model_config.gammas)

        self.beta = model_config.top_beta
        self.aggregator_function = model_config.aggregator
        self.warmup = model_config.warmup
        self.adapt_top_decoder_variance = self.set_top_decoder_variance(model_config)
    
    def set_top_decoder_variance(self, config):
        if config.adapt_top_decoder_variance is None:
            return []
        else :
            for m in config.adapt_top_decoder_variance :
                if m not in self.modalities_name:
                    raise AttributeError(f"A string provided in *adapt_top_decoder_variance* field doesn't match any of the modalities name : {m} is not in {self.modalities_name}")
            return config.adapt_top_decoder_variance

    def set_bottom_betas(self, bottom_betas):
        if bottom_betas is None:
            self.bottom_betas = {m: 1.0 for m in self.encoders}
        else:
            if bottom_betas.keys() != self.encoders.keys():
                raise AttributeError(
                    "The bottom_betas keys do not match the modalities"
                    "names in encoders."
                )
            else:
                self.bottom_betas = bottom_betas

    def set_gammas(self, gammas):
        if gammas is None:
            self.gammas = {m: 1.0 for m in self.encoders}
        else:
            if gammas.keys() != self.encoders.keys():
                raise AttributeError(
                    "The gammas keys do not match the modalities" "names in encoders."
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
                "Please provide encoders architectures or "
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

    def default_top_encoders(self, model_config: NexusConfig):
        if model_config.modalities_specific_dim is None:
            raise AttributeError(
                "Please provide encoders architectures or "
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

    def default_top_decoders(self, model_config: NexusConfig):
        if (
            model_config.input_dims is None
            or model_config.modalities_specific_dim is None
        ):
            raise AttributeError(
                "Please provide encoders architectures or "
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

    def default_joint_encoder(self, model_config: NexusConfig):
        return Encoder_VAE_MLP(
            BaseAEConfig(
                input_dim=(model_config.msg_dim,), latent_dim=model_config.latent_dim
            )
        )

    def set_top_encoders(self, encoders):
        self.top_encoders = nn.ModuleDict()
        for k in encoders:
            if not isinstance(encoders[k], BaseEncoder):
                raise AttributeError(
                    "Top Encoders must be instances of multivae.models.base.BaseEncoder"
                )
            else:
                self.top_encoders[k] = encoders[k]

    def set_top_decoders(self, decoders):
        self.top_decoders = nn.ModuleDict()
        for k in decoders:
            if not isinstance(decoders[k], BaseDecoder):
                raise AttributeError(
                    "Top Decoders must be instances of multivae.models.base.BaseDecoder"
                )
            else:
                self.top_decoders[k] = decoders[k]

    def set_joint_encoder(self, joint_encoder):
        if not isinstance(joint_encoder, BaseEncoder):
            raise AttributeError(
                "Joint encoder must be an instance of multivae.models.base.BaseEncoder"
            )
        else:
            self.joint_encoder = joint_encoder

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        
        epoch = kwargs.pop("epoch", 1)
        annealing = min(epoch/self.warmup, 1.0)
        
        # Compute the first level representations and ELBOs
        modalities_msg = dict()
        first_level_elbos = 0
        first_level_z = dict()

        for m in inputs.data:
            output_m = self.encoders[m](inputs.data[m])
            mu, logvar = output_m.embedding, output_m.log_covariance
            sigma = torch.exp(0.5 * logvar)

            z = dist.Normal(mu, sigma).rsample()

            # re-decode
            recon = self.decoders[m](z).reconstruction

            # Compute the modalities specific ELBOs
            logprob = (
                -(
                    self.recon_log_probs[m](recon, inputs.data[m])
                    * self.rescale_factors[m]
                )
                .reshape(recon.size(0), -1)
                .sum(-1)
            )

            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            elbo = logprob + KLD * self.bottom_betas[m] * annealing

            # Pass the modality specific latent variable through the top encoder to compute the message
            msg = self.top_encoders[m](z.clone().detach()).embedding

            # Use masks to filter out unavailable samples
            if hasattr(inputs, "masks"):
                elbo = elbo * inputs.masks[m].float()
                z = (z.permute(1, 0) * inputs.masks[m].float()).permute(1, 0)
                msg = (msg.permute(1, 0) * inputs.masks[m].float()).permute(1, 0)

            first_level_elbos += elbo

            modalities_msg[m] = msg
            first_level_z[m] = z

        # Aggregate the modalities messages
        aggregated_msg = self.aggregate_msg(inputs, modalities_msg, apply_dropout=True)

        # Compute the higher level latent variable and ELBO
        joint_output = self.joint_encoder(aggregated_msg)
        joint_mu, joint_log_var = joint_output.embedding, joint_output.log_covariance
        joint_sigma = torch.exp(0.5 * joint_log_var)

        joint_z = dist.Normal(joint_mu, joint_sigma).rsample()

        joint_elbo = 0
        for m in self.top_decoders:
            recon = self.top_decoders[m](joint_z).reconstruction

            # Eventually adapt the scale of the top decoder
            if m in self.adapt_top_decoder_variance :
                scale = ((first_level_z[m].clone().detach() - recon) ** 2).mean([0, 1], keepdim=True).sqrt()
            else :
                scale = 1

            
            joint_elbo += -(
                dist.Normal(recon, scale).log_prob(first_level_z[m]) * self.gammas[m]
            ).sum(-1)

        joint_KLD = -0.5 * torch.sum(
            1 + joint_log_var - joint_mu.pow(2) - joint_log_var.exp(), dim=1
        )
        joint_elbo += self.beta * joint_KLD * annealing

        total_loss = joint_elbo + first_level_elbos

        return ModelOutput(loss=total_loss.mean(0), 
                           metrics={'annealing' : annealing,
                                    'joint_elbo' : joint_elbo.mean(0),
                                    'joint_KLD' : joint_KLD.mean(0)})

    def rsample(self, encoder_output: ModelOutput, N=1, flatten=False):
        mu = encoder_output.embedding
        sigma = torch.exp(0.5 * encoder_output.log_covariance)
        shape = [] if N == 1 else [N]

        z = dist.Normal(mu, sigma).rsample(shape)
        if N > 1 and flatten:
            N, l, d = z.shape
            z = z.reshape(l * N, d)
        return z

    def aggregate_msg(
        self, inputs: MultimodalBaseDataset, modalities_msg: dict, apply_dropout=False
    ):
        if self.aggregator_function == "mean":
            # With an already incomplete dataset, we don't apply dropout
            if hasattr(inputs, "masks"):
                normalization_per_sample = torch.stack(
                    [inputs.masks[m] for m in inputs.masks], dim=0
                ).sum(0)
                assert not normalization_per_sample.isnan().any()
                assert not normalization_per_sample.isinf().any()
                assert not (normalization_per_sample == 0.0).any()

                aggregated_msg = torch.sum(
                    torch.stack(list(modalities_msg.values()), dim=0), dim=0
                )

                aggregated_msg = (aggregated_msg.t() / normalization_per_sample).t()

            # With a complete dataset, we apply forced perceptual dropout during training
            else:
                bernoulli_drop = (dist.Bernoulli(self.dropout).sample() == 1).item()

                if apply_dropout and bernoulli_drop:
                    # randomly select the subset
                    subset_size = torch.randint(
                        low=1, high=self.n_modalities, size=(1,)
                    ).item()
                    subset_to_drop = np.random.choice(
                        list(modalities_msg.keys()), size=subset_size, replace=False
                    )

                    remaining_mods = [
                        modalities_msg[m]
                        for m in modalities_msg
                        if m not in subset_to_drop
                    ]
                    aggregated_msg = torch.stack(remaining_mods, dim=0).sum(0) / (
                        self.n_modalities - subset_size
                    )

                else:
                    aggregated_msg = (
                        torch.sum(
                            torch.stack(list(modalities_msg.values()), dim=0), dim=0
                        )
                        / self.n_modalities
                    )

            assert not aggregated_msg.isnan().any()
            assert not aggregated_msg.isinf().any()

            return aggregated_msg

        else:
            raise AttributeError(
                f"The aggregator function {self.aggregator}"
                "is not supported at the moment for the nexus model."
            )

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        """
        This function computes the high level representation of the input modalities. It returns a ModelOutput
        instance 
        
        python```
        return ModelOutput(z = ...,
                           modalities_z = ...,
                           one_latent_space=...)
        ```
        
        It is assumed that for all inputs, the modalities in cond_mod are available.
        

        """

        modalities_z = dict()
        modalities_msg = dict()
        flatten = kwargs.pop("flatten", False)

        # Encode each modality with the bottom encoders
        for m in cond_mod:
            output_m = self.encoders[m](inputs.data[m])
            modalities_z[m] = self.rsample(output_m, N, flatten)
            modalities_msg[m] = self.top_encoders[m](output_m.embedding).embedding

        # Compute high level representation
        aggregated_msg = self.aggregate_msg(inputs, modalities_msg)

        z = self.rsample(self.joint_encoder(aggregated_msg), N=N, flatten=flatten)

        return ModelOutput(z=z, one_latent_space=True, modalities_z=modalities_z)

    def decode(self, embedding: ModelOutput, modalities: Union[list,str] = "all", **kwargs):
        self.eval()

        if modalities == "all":
            modalities = list(self.encoders.keys())

        use_bottom_z_for_reconstruction = kwargs.pop("use_bottom_z_for_recon", False)

        outputs = ModelOutput()

        for m in modalities:
            if (use_bottom_z_for_reconstruction) and (
                m in embedding.modalities_z.keys()
            ):
                z_m = embedding.modalities_z[m]
            else:
                z_m = self.top_decoders[m](embedding.z).reconstruction

            outputs[m] = self.decoders[m](z_m).reconstruction

        return outputs
