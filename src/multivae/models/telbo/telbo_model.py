from typing import Dict, Union

import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from ..joint_models import BaseJointModel
from .telbo_config import TELBOConfig


class TELBO(BaseJointModel):

    """
    The Triple ELBO VAE model.

    Args:

        model_config (TELBOConfig): An instance of TELBOConfig in which any model's parameters is
            made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoder (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.

        joint_encoder (~pythae.models.nn.base_architectures.BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.


    """

    def __init__(
        self,
        model_config: TELBOConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        self.model_name = "TELBO"
        self.warmup = model_config.warmup
        self.reset_optimizer_epochs = [self.warmup]

        if model_config.lambda_factors is None:
            self.lambda_factors = self.rescale_factors
        else:
            self.lambda_factors = model_config.lambda_factors
        if model_config.gamma_factors is None:
            self.gamma_factors = self.rescale_factors
        else:
            self.gamma_factors = model_config.gamma_factors

    def _set_torch_no_grad_on_joint_vae(self):
        """Function used to freeze the parameters of the joint
        encoder and decoders after the warmup."""
        self.joint_encoder.requires_grad_(False)
        self.decoders.requires_grad_(False)

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        epoch = kwargs.pop("epoch", 1)

        # First compute the joint ELBO
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)
        z_joint = qz_xy.rsample()

        recon_loss = 0

        # Decode in each modality
        len_batch = 0
        for mod in self.decoders:
            x_mod = inputs.data[mod]
            len_batch = len(x_mod)
            recon_mod = self.decoders[mod](z_joint).reconstruction
            recon_loss += (
                -self.recon_log_probs[mod](recon_mod, x_mod) * self.lambda_factors[mod]
            ).sum()

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if epoch <= self.warmup:
            return ModelOutput(
                recon_loss=recon_loss / len_batch,
                KLD=KLD / len_batch,
                loss=(recon_loss + KLD) / len_batch,
                metrics=dict(kld_joint=KLD, recon_joint=recon_loss / len_batch),
            )

        else:
            self._set_torch_no_grad_on_joint_vae()
            mod_elbos = {}
            loss = 0

            for mod in self.encoders:
                mod_output = self.encoders[mod](inputs.data[mod])
                mod_mu, mod_log_var = mod_output.embedding, mod_output.log_covariance

                mod_sigma = torch.exp(0.5 * mod_log_var)
                qz_x0 = dist.Normal(mod_mu, mod_sigma)
                mod_z = qz_x0.rsample()

                mod_recon = self.decoders[mod](mod_z).reconstruction
                mod_recon_loss = (
                    -self.recon_log_probs[mod](mod_recon, inputs.data[mod])
                    * self.gamma_factors[mod]
                )

                mod_kld = -0.5 * torch.sum(
                    1 + log_var - mod_mu.pow(2) - mod_log_var.exp()
                )
                mod_elbos[mod] = mod_recon_loss.sum() + mod_kld
                loss += mod_recon_loss.sum() + mod_kld

            return ModelOutput(loss=loss / len_batch, metrics=mod_elbos)

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ) -> ModelOutput:
        self.eval()

        if type(cond_mod) == list and len(cond_mod) == 1:
            cond_mod = cond_mod[0]

        if cond_mod == "all" or (
            type(cond_mod) == list and len(cond_mod) == self.n_modalities
        ):
            output = self.joint_encoder(inputs.data)
            sample_shape = [] if N == 1 else [N]
            z = dist.Normal(
                output.embedding, torch.exp(0.5 * output.log_covariance)
            ).rsample(sample_shape)
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)

        if type(cond_mod) == list and len(cond_mod) != 1:
            raise AttributeError(
                "Conditioning on a subset containing more than one modality "
                "is not yet implemented."
            )

        if cond_mod in self.modalities_name:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            sample_shape = [] if N == 1 else [N]

            z = dist.Normal(
                output.embedding, torch.exp(0.5 * output.log_covariance)
            ).rsample(sample_shape)

            if N > 1 and kwargs.pop("flatten", False):
                z = z.reshape(-1, self.latent_dim)

            return ModelOutput(z=z, one_latent_space=True)
        else:
            raise AttributeError(
                f"Modality of name {cond_mod} not handled. The"
                f" modalities that can be encoded are {list(self.encoders.keys())}"
            )
