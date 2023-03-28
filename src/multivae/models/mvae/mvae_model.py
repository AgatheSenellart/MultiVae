from itertools import combinations
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from numpy.random import choice
from pythae.models.base.base_utils import ModelOutput
from scipy.special import comb
from torch.distributions import kl_divergence

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

        self.k = model_config.k
        if self.n_modalities <= 2:
            self.k = 0
        self.set_subsets()
        self.warmup = model_config.warmup
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

        joint_std = torch.exp(0.5 * lnV)
        return joint_mu, joint_std

    def _compute_mu_logvar_subset(self, data: dict, subset: list):
        """Computes the parameters of the posterior when conditioning on
        the modalities contained in subset."""
        mus_sub = []
        log_vars_sub = []
        for mod in self.encoders:
            if mod in subset:
                output_mod = self.encoders[mod](data[mod])
                mu_mod, log_var_mod = output_mod.embedding, output_mod.log_covariance
                mus_sub.append(mu_mod)
                log_vars_sub.append(log_var_mod)
        sub_mu, sub_log_var = self.poe(mus_sub, log_vars_sub)
        return sub_mu, sub_log_var

    def _compute_elbo_subset(self, data: dict, subset: list, beta: float):
        len_batch = 0
        sub_mu, sub_log_var = self._compute_mu_logvar_subset(data, subset)
        z = dist.Normal(sub_mu, torch.exp(0.5 * sub_log_var)).rsample()
        elbo_sub = 0
        for mod in self.decoders:
            if mod in subset:
                len_batch = len(data[mod])
                recon = self.decoders[mod](z).reconstruction
                elbo_sub += -self.recon_log_probs[mod](recon, data[mod]).sum()
        elbo_sub += self.kl_prior(sub_mu, torch.exp(0.5 * sub_log_var)) * beta
        return elbo_sub / len_batch

    def _filter_inputs_with_masks(
        self, inputs: IncompleteDataset, subset: Union[list, tuple]
    ):
        """Returns a filtered dataset containing only the samples that are available
        in all the modalities contained in subset."""

        filter = torch.tensor(
            True,
        ).to(inputs.masks[subset[0]].device)
        for mod in subset:
            filter = torch.logical_and(filter, inputs.masks[mod])

        filtered_inputs = {}
        for mod in subset:
            print(filter, inputs.data[mod])
            filtered_inputs[mod] = inputs.data[mod][filter]
        return filtered_inputs, filter

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
        if epoch >= self.warmup:
            beta = 1
        else:
            beta = epoch / self.warmup

        total_loss = 0
        # Collect all the subsets
        # Add the unimodal subset
        subsets = [[m] for m in self.encoders]
        # Add the joint subset
        subsets.append([m for m in self.encoders])
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
            else:
                filtered_inputs = inputs.data
            total_loss += self._compute_elbo_subset(filtered_inputs, s, beta)

        return ModelOutput(loss=total_loss, metrics=dict())

    def kl_prior(self, mu, std):
        return kl_divergence(dist.Normal(mu, std), dist.Normal(0, 1)).sum()

    def encode(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        # If the input cond_mod is a string : convert it to a list
        if type(cond_mod) == str:
            if cond_mod == "all":
                cond_mod = list(self.encoders.keys())
            elif cond_mod in self.encoders.keys():
                cond_mod = [cond_mod]
            else:
                raise AttributeError(
                    'If cond_mod is a string, it must either be "all" or a modality name'
                    f" The provided string {cond_mod} is neither."
                )

        # Check that all data is available for the desired conditioned modalities
        if hasattr(inputs, "masks"):
            _, filter = self._filter_inputs_with_masks(inputs, cond_mod)
            if not filter.all():
                raise AttributeError(
                    "You asked to encode conditioned on the following"
                    f"modalities {cond_mod} but some of the modalities are missing in the input data"
                    " (according to the provided masks)"
                )

        sub_mu, sub_log_var = self._compute_mu_logvar_subset(inputs.data, cond_mod)
        sample_shape = [N] if N > 1 else []
        z = dist.Normal(sub_mu, torch.exp(0.5 * sub_log_var)).rsample(sample_shape)
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
        # Only keep the complete samples
        all_modalities = list(self.encoders.keys())
        if hasattr(inputs, "masks"):
            filtered_inputs, filter = self._filter_inputs_with_masks(
                inputs, all_modalities
            )
        else:
            filtered_inputs = inputs.data

        

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        n_data = len(filtered_inputs[list(filtered_inputs.keys())[0]])
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            
            # Compute the parameters of the joint posterior
            mu, log_var = self._compute_mu_logvar_subset({k:filtered_inputs[k][i] for k in filtered_inputs},
                                                         all_modalities)
            assert(mu.shape==(1,self.latent_dim))
            sigma = torch.exp(0.5 * log_var)
            qz_xy = dist.Normal(mu, sigma)
            # And sample from the posterior
            z_joint = qz_xy.rsample([K]).squeeze()  # shape K x latent_dim
            print(z_joint.shape)

            
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

                    dim_reduce = tuple(range(1, len(recon.shape)))
                    lpx_zs += self.recon_log_probs[mod](recon, x_m).sum(dim=dim_reduce)

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

        return -ll / n_data
