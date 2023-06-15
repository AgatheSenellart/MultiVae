import logging
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from ..joint_models import BaseJointModel
from .jmvae_config import JMVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class JMVAE(BaseJointModel):

    """The Joint Multimodal Variational Autoencoder model.

    Args:
        model_config (JMVAEConfig): An instance of JMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoder (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.

        joint_encoder (~pythae.models.nn.base_architectures.BaseEncoder) : An instance of
            BaseEncoder that takes all the modalities as an input. If none is provided, one is
            created from the unimodal encoders. Default : None.
    """

    def __init__(
        self,
        model_config: JMVAEConfig,
        encoders: dict = None,
        decoders: dict = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        self.model_name = "JMVAE"

        self.alpha = model_config.alpha
        self.warmup = model_config.warmup

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        """Compute encodings from the posterior distributions conditioning on all the modalities or
        a subset of the modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.
            cond_mod (Union[list, str], optional): The modalities to use to compute the posterior
            distribution. Defaults to 'all'.
            N (int, optional): The number of samples to generate from the posterior distribution
            for each datapoint. Defaults to 1.

        Raises:
            AttributeError: _description_
            AttributeError: _description_

        Returns:
            ModelOutput: Containing z the latent variables.
        """
        self.eval()

        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod
        sample_shape = [] if N == 1 else [N]
        return_mean = kwargs.pop("return_mean", False)

        if len(cond_mod) == self.n_modalities:
            output = self.joint_encoder(inputs.data)
            if return_mean:
                z = torch.stack([output.embedding] * N) if N > 1 else output.embedding
            else:
                z = dist.Normal(
                    output.embedding, torch.exp(0.5 * output.log_covariance)
                ).rsample(sample_shape)
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)

        elif len(cond_mod) != 1:
            z = self.sample_from_poe_subset_exact(
                cond_mod, inputs.data, sample_shape=sample_shape
            )

            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)

        elif len(cond_mod) == 1:
            cond_mod = cond_mod[0]
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            sample_shape = [] if N == 1 else [N]

            z = dist.Normal(
                output.embedding, torch.exp(0.5 * output.log_covariance)
            ).rsample(
                sample_shape
            )  # shape N x len_data x latent_dim
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)
        else:
            raise AttributeError(
                f"Modalities of names {cond_mod} not handled. The"
                f" modalities that can be encoded are {list(self.encoders.keys())}"
            )

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Performs a forward pass of the JMVAE model on inputs.

        Args:
            inputs (MultimodalBaseDataset)
            warmup (int) : number of warmup epochs to do. The weigth of the regularization augments
                linearly to reach 1 at the end of the warmup. The enforces the optimization of
                the reconstruction term only at first.
            epoch (int) : the epoch number during which forward is called.

        Returns:
            ModelOutput
        """

        epoch = kwargs.pop("epoch", 1)

        # Compute the reconstruction term
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
                -self.recon_log_probs[mod](recon_mod, x_mod) * self.rescale_factors[mod]
            ).sum()

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Compute the KL between unimodal and joint encoders
        LJM = 0
        for mod in self.encoders:
            output = self.encoders[mod](inputs.data[mod])
            uni_mu, uni_log_var = output.embedding, output.log_covariance
            LJM += (
                1
                / 2
                * (
                    uni_log_var
                    - log_var
                    + (torch.exp(log_var) + (mu - uni_mu) ** 2) / torch.exp(uni_log_var)
                    - 1
                )
            )

        LJM = LJM.sum() * self.alpha

        # Compute the total loss to minimize

        reg_loss = KLD + LJM
        if epoch >= self.warmup:
            beta = 1
        else:
            beta = epoch / self.warmup
        recon_loss, reg_loss = recon_loss / len_batch, reg_loss / len_batch
        elbo = (recon_loss + KLD) / len_batch
        loss = recon_loss + beta * reg_loss

        metrics = dict(loss_no_ponderation=reg_loss + recon_loss, beta=beta, elbo=elbo)

        output = ModelOutput(loss=loss, metrics=metrics)

        return output

    def poe(self, mus_list, log_vars_list):
        mus = mus_list.copy()
        log_vars = log_vars_list.copy()

        # Add the prior to the product of experts
        mus.append(torch.zeros_like(mus[0]))
        log_vars.append(torch.zeros_like(log_vars[0]))

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars])  # Compute the inverse of variances
        lnV = -torch.logsumexp(lnT, dim=0)  # variances of the product of expert
        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        return joint_mu, lnV

    def sample_from_poe_subset_exact(self, subset: list, data: dict, sample_shape=[]):
        """
        Sample from the product of experts for infering from a subset of modalities.
        """

        mus, logvars = [], []

        for mod in subset:
            vae_output = self.encoders[mod](data[mod])
            mu, log_var = vae_output.embedding, vae_output.log_covariance
            mus.append(vae_output.embedding)
            logvars.append(vae_output.log_covariance)

        joint_mu, joint_logvar = self.poe(mus, logvars)
        z = dist.Normal(joint_mu, torch.exp(0.5 * joint_logvar)).rsample(sample_shape)
        return z

    def compute_joint_nll_paper(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """
        Return the estimated negative log-likelihood summed over the input batch.
        The negative log-likelihood is estimated using importance sampling.

        Args:
            inputs : the data to compute the joint likelihood"""

        # First compute all the parameters of the joint posterior q(z|x,y)

        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)
        # And sample from the posterior
        z_joint = qz_xy.rsample([K])  # shape K x n_data x latent_dim
        z_joint = z_joint.permute(1, 0, 2)
        n_data, _, latent_dim = z_joint.shape

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0

        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            while start_idx < stop_idx:
                latents = z_joint[i][start_idx:stop_idx]

                # Compute p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0  # ln(p(x,y|z))
                for mod in ["images"]:  # only keep the images for this likelihood
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, *decoder_output_shape)
                    x_m = inputs.data[mod][i]  # (*input_shape)
                    lpx_zs += (
                        self.recon_log_probs[mod](recon, x_m)
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

                # Compute ln(p(z))
                prior = dist.Normal(0, 1)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                qz_xy = dist.Normal(mu[i], sigma[i])
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(start_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
