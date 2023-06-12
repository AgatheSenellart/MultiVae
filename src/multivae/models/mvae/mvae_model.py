from itertools import combinations
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from numpy.random import choice
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from .mvae_config import MVAEConfig


class MVAE(BaseMultiVAE):
    """
    The Multi-modal VAE model.

    Args:
        model_config (MVAEConfig): An instance of MVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoder (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.
    """

    def __init__(
        self, model_config: MVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.subsampling = model_config.use_subsampling
        self.k = model_config.k
        if self.n_modalities <= 2:
            self.k = 0
        self.set_subsets()
        self.warmup = model_config.warmup
        self.beta = model_config.beta
        self.model_name = "MVAE"

    def set_subsets(self):
        self.subsets = []
        for i in range(2, self.n_modalities):
            self.subsets += combinations(list(self.encoders.keys()), r=i)

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

    # def poe(self, mus_list, logvar_list, eps=1e-8):

    # ORIGINAL VERSION BUT LESS STABLE

    #     mus = mus_list.copy()
    #     log_vars = logvar_list.copy()

    #     # Add the prior to the product of experts
    #     mus.append(torch.zeros_like(mus[0]))
    #     log_vars.append(torch.zeros_like(log_vars[0]))

    #     mus = torch.stack(mus)
    #     logvars = torch.stack(log_vars)
    #     var       = torch.exp(logvars) + eps
    #     # precision of i-th Gaussian expert at point x
    #     T         = 1. / (var + eps)
    #     pd_mu     = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
    #     pd_var    = 1. / torch.sum(T, dim=0)
    #     pd_logvar = torch.log(pd_var + eps)
    #     return pd_mu, pd_logvar

    def compute_mu_log_var_subset(self, inputs: MultimodalBaseDataset, subset: list):
        """Computes the parameters of the posterior when conditioning on
        the modalities contained in subset."""
        mus_sub = []
        log_vars_sub = []
        for mod in self.encoders:
            if mod in subset:
                output_mod = self.encoders[mod](inputs.data[mod])
                mu_mod, log_var_mod = output_mod.embedding, output_mod.log_covariance
                if hasattr(inputs, "masks"):
                    log_var_mod[
                        (1 - inputs.masks[mod].int()).bool().flatten()
                    ] = torch.inf

                mus_sub.append(mu_mod)
                log_vars_sub.append(log_var_mod)
        sub_mu, sub_logvar = self.poe(mus_sub, log_vars_sub)
        return sub_mu, sub_logvar

    def _compute_elbo_subset(
        self, inputs: MultimodalBaseDataset, subset: list, beta: float
    ):
        sub_mu, sub_logvar = self.compute_mu_log_var_subset(inputs, subset)
        sub_std = torch.exp(0.5 * sub_logvar)
        z = dist.Normal(sub_mu, sub_std).rsample()
        elbo_sub = 0
        for mod in self.decoders:
            if mod in subset:
                recon = self.decoders[mod](z).reconstruction
                recon_mod = (
                    -(
                        self.recon_log_probs[mod](recon, inputs.data[mod])
                        * self.rescale_factors[mod]
                    )
                    .reshape(recon.size(0), -1)
                    .sum(-1)
                )

                if hasattr(inputs, "masks"):
                    recon_mod = inputs.masks[mod].float() * recon_mod
                elbo_sub += recon_mod.sum()

        recon = elbo_sub
        KLD = -0.5 * torch.sum(1 + sub_logvar - sub_mu.pow(2) - sub_logvar.exp())
        elbo_sub += KLD * beta

        return elbo_sub / len(sub_mu), KLD / len(sub_mu), recon / len(sub_mu)

    def _filter_inputs_with_masks(
        self, inputs: IncompleteDataset, subset: Union[list, tuple]
    ):
        """Returns a filtered dataset containing only the samples that are available
        in at least one of the modalities contained in subset."""

        filter = torch.tensor(
            False,
        ).to(inputs.masks[subset[0]].device)
        for mod in subset:
            filter = torch.logical_or(filter, inputs.masks[mod])

        filtered_inputs = {}
        filtered_masks = {}

        for mod in subset:
            filtered_inputs[mod] = inputs.data[mod][filter]
            filtered_masks[mod] = inputs.masks[mod][filter]

        filtered_dataset = IncompleteDataset(data=filtered_inputs, masks=filtered_masks)
        return filtered_dataset, filter

    def forward(
        self, inputs: Union[MultimodalBaseDataset, IncompleteDataset], **kwargs
    ):
        """The main function of the model that computes the loss and some monitoring metrics.
        One of the advantages of MVAE is that we can train with incomplete data.

        Args:
            inputs (MultimodalBaseDataset): The data. It can be an instance of IncompleteDataset
                which contains a field masks for weakly supervised learning.
                masks is a dictionary indicating which datasamples are missing in each of the modalities.
                For each modality, a boolean tensor indicates which samples are available. (The non
                available samples are assumed to be replaced with zero values in the multimodal dataset entry.)
        """

        epoch = kwargs.pop("epoch", 1)
        batch_ratio = kwargs.pop("batch_ratio", 0)
        if epoch >= self.warmup:
            beta = 1 * self.beta
        else:
            beta = (epoch - 1 + batch_ratio) / self.warmup * self.beta
        total_loss = 0
        metrics = {}
        # Collect all the subsets
        subsets = []
        # Add the joint subset
        subsets.append([m for m in self.encoders])
        if self.subsampling:
            # Add the unimodal subsets
            subsets.extend([[m] for m in self.encoders])
            # Add random subsets
            if self.k > 0:
                random_idx = choice(
                    np.arange(len(self.subsets)), size=self.k, replace=False
                )
                for id in random_idx:
                    subsets.append(self.subsets[id])

        for s in subsets:
            if hasattr(inputs, "masks"):
                filtered_inputs, filter = self._filter_inputs_with_masks(inputs, s)
                not_all_samples_missing = torch.any(filter)
            else:
                filtered_inputs = inputs
                not_all_samples_missing = True

            if not_all_samples_missing:
                subset_elbo, subset_kld, subset_recon = self._compute_elbo_subset(
                    filtered_inputs, s, beta
                )
            else:
                subset_elbo = 0
            total_loss += subset_elbo
            metrics["_".join(sorted(s))] = subset_elbo
            metrics["beta"] = beta
            metrics["kld" + "_".join(sorted(s))] = subset_kld
            metrics["recon" + "_".join(sorted(s))] = subset_recon

        return ModelOutput(loss=total_loss, metrics=metrics)

    def encode(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        sub_mu, sub_logvar = self.compute_mu_log_var_subset(inputs, cond_mod)
        sub_std = torch.exp(0.5 * sub_logvar)
        sample_shape = [N] if N > 1 else []

        return_mean = kwargs.pop("return_mean", False)

        if return_mean:
            z = torch.stack([sub_mu] * N) if N > 1 else sub_mu
        else:
            z = dist.Normal(sub_mu, sub_std).rsample(sample_shape)
        flatten = kwargs.pop("flatten", False)
        if flatten:
            z = z.reshape(-1, self.latent_dim)

        return ModelOutput(z=z, one_latent_space=True)

    def compute_joint_nll(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Computes the joint_negative_nll for a batch of inputs."""

        # iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        n_data = len(inputs.data[list(inputs.data.keys())[0]])
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []

            # Compute the parameters of the joint posterior
            mu, log_var = self.compute_mu_log_var_subset(
                MultimodalBaseDataset(
                    data={k: inputs.data[k][i].unsqueeze(0) for k in inputs.data}
                ),
                list(self.encoders.keys()),
            )

            sigma = torch.exp(0.5 * log_var)
            qz_xy = dist.Normal(mu, sigma)

            # And sample from the posterior
            z_joint = qz_xy.rsample([K]).squeeze()  # shape K x latent_dim

            while start_idx < stop_idx:
                latents = z_joint[start_idx:stop_idx]

                # Compute p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0  # ln(p(x,y|z))
                for mod in inputs.data:
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, nb_channels, w, h)
                    x_m = inputs.data[mod][i]  # (nb_channels, w, h)

                    lpx_zs += (
                        self.recon_log_probs[mod](
                            recon, torch.stack([x_m] * len(recon))
                        )
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

                # Compute ln(p(z))
                prior = dist.Normal(0, 1)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                qz_xy = dist.Normal(mu.squeeze(), sigma.squeeze())
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
