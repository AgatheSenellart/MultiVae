import logging
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from torch.distributions import Laplace, Normal

from multivae.data.datasets.base import MultimodalBaseDataset

from ..base import BaseMultiVAE
from .mmvae_config import MMVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MMVAE(BaseMultiVAE):

    """
    The Variational Mixture-of-Experts Autoencoder model.

    Args:
        model_config (MMVAEConfig): An instance of MMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.
    """

    def __init__(
        self, model_config: MMVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.K = model_config.K
        if model_config.prior_and_posterior_dist == "laplace_with_softmax":
            self.post_dist = Laplace
            self.prior_dist = Laplace
        elif model_config.prior_and_posterior_dist == "normal":
            self.post_dist = Normal
            self.prior_dist = Normal
        else:
            raise AttributeError(
                " The posterior_dist parameter must be "
                " either 'laplace_with_softmax' or 'normal'. "
                f" {model_config.prior_and_posterior_dist} was provided."
            )

        self.prior_mean = torch.nn.Parameter(
            torch.zeros(1, self.latent_dim), requires_grad=False
        )
        self.prior_log_var = torch.nn.Parameter(
            torch.zeros(1, self.latent_dim), requires_grad=model_config.learn_prior
        )

        self.model_name = "MMVAE"

    def log_var_to_std(self, log_var):
        """
        For latent distributions parameters, transform the log covariance to the
        standard deviation of the distribution either applying softmax or not.
        This follows the original implementation.
        """

        if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
            return F.softmax(log_var, dim=-1) * log_var.size(-1) + 1e-6
        else:
            return torch.exp(0.5 * log_var)

    @property
    def pz_params(self):
        """From the prior mean and log_covariance, return the mean and standard
        deviation, either applying softmax or not depending on the choice of prior
        distribution.

        Returns:
            tuple: mean, std
        """
        mean = self.prior_mean
        if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
            std = (
                F.softmax(self.prior_log_var, dim=-1) * self.prior_log_var.size(-1)
                + 1e-6
            )
        else:
            std = torch.exp(0.5 * self.prior_log_var)
        return mean, std

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        # TODO : maybe implement a minibatch strategy for stashing the gradients before
        # backpropagation when using a large number k.
        # Also, I've only implemented the dreg_looser loss but it may be nice to offer other options.

        # First compute all the encodings for all modalities
        embeddings = {}
        qz_xs = {}
        qz_xs_detach = {}
        reconstructions = {}

        compute_loss = kwargs.pop("compute_loss", True)
        detailed_output = kwargs.pop("detailed_output", False)
        K = kwargs.pop("K", self.K)

        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance

            sigma = self.log_var_to_std(log_var)

            qz_x = self.post_dist(mu, sigma)
            z_x = qz_x.rsample([K])

            # The DREG loss uses detached parameters in the loss computation afterwards.
            qz_x_detach = self.post_dist(mu.detach(), sigma.detach())

            # Then compute all the cross-modal reconstructions
            reconstructions[cond_mod] = {}
            for recon_mod in self.decoders:
                decoder = self.decoders[recon_mod]
                recon = decoder(z_x)["reconstruction"]

                reconstructions[cond_mod][recon_mod] = recon

            qz_xs[cond_mod] = qz_x
            embeddings[cond_mod] = z_x
            qz_xs_detach[cond_mod] = qz_x_detach

        # Compute DREG loss
        if compute_loss:
            # TODO : change
            loss_output = self.dreg_looser(
                qz_xs_detach, embeddings, reconstructions, inputs
            )

        else:
            loss_output = ModelOutput()
        if detailed_output:
            loss_output["qz_xs"] = qz_xs
            loss_output["qz_xs_detach"] = qz_xs_detach
            loss_output["zss"] = embeddings
            loss_output["recon"] = reconstructions

        return loss_output

    def dreg_looser(self, qz_xs, embeddings, reconstructions, inputs):
        if hasattr(inputs, "masks"):
            # Compute the number of available modalities per sample
            n_mods_sample = torch.sum(
                torch.stack(tuple(inputs.masks.values())).int(), dim=0
            )
        else:
            n_mods_sample = torch.tensor([self.n_modalities])

        lws = []
        zss = []
        for mod in embeddings:
            z = embeddings[mod]  # (K, n_batch, latent_dim)
            n_mods_sample = n_mods_sample.to(z.device)
            prior = self.prior_dist(*self.pz_params)
            lpz = prior.log_prob(z).sum(-1)

            if hasattr(inputs, "masks"):
                lqz_x = torch.stack(
                    [
                        qz_xs[m].log_prob(z).sum(-1) * inputs.masks[m].float()
                        for m in qz_xs
                    ]
                )  # n_modalities,K,nbatch
            else:
                lqz_x = torch.stack(
                    [qz_xs[m].log_prob(z).sum(-1) for m in qz_xs]
                )  # n_modalities,K,nbatch

            lqz_x = torch.logsumexp(lqz_x, dim=0) - torch.log(
                n_mods_sample
            )  # log_mean_exp
            lpx_z = 0
            for recon_mod in reconstructions[mod]:
                x_recon = reconstructions[mod][recon_mod]
                K, n_batch = x_recon.shape[0], x_recon.shape[1]
                lpx_z_mod = (
                    self.recon_log_probs[recon_mod](x_recon, inputs.data[recon_mod])
                    .view(K, n_batch, -1)
                    .mul(self.rescale_factors[recon_mod])
                    .sum(-1)
                )

                if hasattr(inputs, "masks"):
                    # cancel unavailable modalities
                    lpx_z_mod *= inputs.masks[recon_mod].float()

                lpx_z += lpx_z_mod

            lw = lpx_z + lpz - lqz_x

            if hasattr(inputs, "masks"):
                # cancel unavailable modalities
                lw *= inputs.masks[mod].float()

            lws.append(lw)
            zss.append(z)

        lws = torch.stack(lws)  # (n_modalities, K, n_batch)
        zss = torch.stack(zss)  # (n_modalities, K, n_batch,latent_dim)
        with torch.no_grad():
            grad_wt = (lws - torch.logsumexp(lws, 1, keepdim=True)).exp()
            if zss.requires_grad:  # True except when we are in eval mode
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)

        lws = (grad_wt * lws).sum(0) / n_mods_sample  # mean over modalities

        return ModelOutput(loss=-lws.sum(), metrics=dict(mean_loss_batch=-lws.mean()))

    def iwae(self, qz_xs, zss, reconstructions, inputs):
        lw_mod = []
        for cond_mod in zss:
            lpz = self.prior_dist(*self.pz_params).log_prob(zss[cond_mod]).sum(-1)
            lqz_x = torch.stack(
                [qz_xs[m].log_prob(zss[cond_mod]).sum(-1) for m in qz_xs]
            )
            lqz_x = torch.logsumexp(lqz_x, dim=0) - np.log(lqz_x.size(0))
            lpx_z = 0
            for recon_mod in reconstructions[cond_mod]:
                x_recon = reconstructions[cond_mod][recon_mod]
                K, n_batch = x_recon.shape[0], x_recon.shape[1]
                lpx_z += (
                    self.recon_log_probs[recon_mod](x_recon, inputs.data[recon_mod])
                    .view(K, n_batch, -1)
                    .mul(self.rescale_factors[recon_mod])
                    .sum(-1)
                )
            lw = lpx_z + lpz - lqz_x  # n_samples , n_batch
            lw_mod.append(lw)

        lw = torch.cat(lw_mod, dim=0)  # (n_modalities* K, n_batch)
        lw = torch.logsumexp(lw, dim=0) - np.log(lw.size(0))
        return ModelOutput(loss=-lw.sum(), metrics=dict())

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        return_mean = kwargs.pop("return_mean", False)
        if all([s in self.encoders.keys() for s in cond_mod]):
            if return_mean:
                emb = torch.stack(
                    [self.encoders[mod](inputs.data[mod]).embedding for mod in cond_mod]
                ).mean(0)
                if N > 1:
                    z = torch.stack([emb] * N)
                else:
                    z = emb

            else:
                # Choose one of the conditioning modalities at random
                mod = np.random.choice(cond_mod)

                output = self.encoders[mod](inputs.data[mod])

                mu, log_var = output.embedding, output.log_covariance
                sigma = self.log_var_to_std(log_var)
                qz_x = self.post_dist(mu, sigma)
                sample_shape = torch.Size([]) if N == 1 else torch.Size([N])
                z = qz_x.rsample(sample_shape)

            flatten = kwargs.pop("flatten", False)
            if flatten:
                z = z.reshape(-1, self.latent_dim)

            return ModelOutput(z=z, one_latent_space=True)

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """
        Return the estimated negative log-likelihood summed over the inputs.
        The negative log-likelihood is estimated using importance sampling.

        Args:
            inputs : the data to compute the joint likelihood

        """

        self.eval()

        # First compute all the parameters of the joint posterior q(z|x,y)
        post_params = []
        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance
            sigma = self.log_var_to_std(log_var)
            post_params.append((mu, sigma))

        z_joint = self.encode(inputs, N=K).z
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
                for mod in inputs.data:
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, nb_channels, w, h)
                    x_m = inputs.data[mod][i]  # (nb_channels, w, h)

                    lpx_zs += (
                        self.recon_log_probs[mod](recon, x_m)
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

                # Compute ln(p(z))
                prior = self.prior_dist(*self.pz_params)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                qz_xs = [self.post_dist(p[0][i], p[1][i]) for p in post_params]
                lqz_xy = torch.logsumexp(
                    torch.stack([q.log_prob(latents).sum(-1) for q in qz_xs]), dim=0
                ) - np.log(self.n_modalities)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll

    @torch.no_grad()
    def compute_joint_nll_paper(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 10
    ):
        """Computes the joint likelihood like in the original dataset, using all Mixture of experts
        samples and modality rescaling."""

        self.eval()

        lws = []
        nb_computed_samples = 0
        while nb_computed_samples < K:
            n_samples = min(batch_size_K, K - nb_computed_samples)
            nb_computed_samples += n_samples
            # Compute a iwae likelihood estimate using n_samples
            output = self.forward(
                inputs, compute_loss=False, K=n_samples, detailed_output=True
            )
            lw = self.iwae(output.qz_xs, output.zss, output.recon, inputs).loss
            lws.append(lw + np.log(n_samples * self.n_modalities))

        ll = torch.logsumexp(torch.stack(lws), dim=0) - np.log(
            nb_computed_samples * self.n_modalities
        )  # n_batch
        return -ll

    def generate_from_prior(self, n_samples, **kwargs):
        sample_shape = [n_samples] if n_samples > 1 else []
        z = self.prior_dist(*self.pz_params).rsample(sample_shape).to(self.device)
        return ModelOutput(z=z.squeeze(), one_latent_space=True)
