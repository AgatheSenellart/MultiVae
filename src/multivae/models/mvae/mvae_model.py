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
                elbo_sub += self.recon_losses[mod](recon, data[mod]).sum()
        elbo_sub += self.kl_prior(sub_mu, torch.exp(0.5 * sub_log_var)) * beta
        return elbo_sub / len_batch

    def _filter_inputs_with_masks(
        self, inputs: IncompleteDataset, subset: Union[list, tuple]
    ):
        """Returns a filtered dataset containing only the samples that are available
        in all the modalities contained in subset."""

        filter = torch.tensor(
            True,
        )
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

    # def infer_latent_from_mod(self, cond_mod, x):
    #     o = self.vaes[cond_mod].encoder(x)
    #     mu, log_var = o.embedding, o.log_covariance
    #     # poe with prior
    #     mu, std = self.poe([mu],[log_var])
    #     z = dist.Normal(mu, std).rsample()
    #     return z

    # def forward(self, x):
    #     """
    #         Using encoders and decoders from both distributions, compute all the latent variables,
    #         reconstructions...
    #     """

    #     # Compute the reconstruction terms
    #     elbo = 0

    #     mus_tilde = []
    #     lnV_tilde = []

    #     for m, vae in enumerate(self.vaes):
    #         o = vae.encoder(x[m])
    #         u_mu, u_log_var = o.embedding, o.log_covariance
    #         # Save the unimodal embedding
    #         mus_tilde.append(u_mu)
    #         lnV_tilde.append(u_log_var)

    #         # Compute the unimodal elbos
    #         mu, std =  self.poe([u_mu], [u_log_var])
    #         # print(m, mu, std)
    #         # mu, std = u_mu, torch.exp(0.5*u_log_var)
    #         z = dist.Normal(mu, std).rsample()
    #         recon = vae.decoder(z).reconstruction
    #         elbo += -1/2*torch.sum((x[m]-recon)**2) * self.lik_scaling[m] - self.kl_prior(mu, std)

    #     # Add the joint elbo
    #     joint_mu, joint_std = self.poe(mus_tilde, lnV_tilde)
    #     z_joint = dist.Normal(joint_mu, joint_std).rsample()

    #     # Reconstruction term in each modality
    #     for m, vae in enumerate(self.vaes):
    #         recon = (vae.decoder(z_joint)['reconstruction'])
    #         elbo += -1/2*torch.sum((x[m]-recon)**2) * self.lik_scaling[m]

    #     # Joint KL divergence
    #     elbo -= self.kl_prior(joint_mu, joint_std)

    #     # If using the subsampling paradigm, sample subsets and compute the poe
    #     if self.subsampling :
    #         # randomly select k subsets
    #         subsets = self.subsets[np.random.choice(len(self.subsets), self.k_subsample,replace=False)]
    #         # print(subsets)

    #         for s in subsets:
    #             sub_mus, sub_log_vars = [mus_tilde[i] for i in s], [lnV_tilde[i] for i in s]
    #             mu, std = self.poe(sub_mus, sub_log_vars)
    #             sub_z = dist.Normal(mu, std).rsample()
    #             elbo -= self.kl_prior(mu, std)
    #             # Reconstruction terms
    #             for m in s:
    #                 recon = self.vaes[m].decoder(sub_z).reconstruction
    #                 elbo += torch.sum(-1 / 2 * (recon - x[m]) ** 2) * self.lik_scaling[m]

    #         # print('computed subsampled elbos')

    #     res_dict = dict(
    #         elbo=elbo,
    #         z_joint=z_joint,
    #         joint_mu=joint_mu,
    #         joint_std=joint_std
    #     )

    #     return res_dict
