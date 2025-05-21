from itertools import chain, combinations
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models.nn.default_architectures import (
    BaseDictDecodersMultiLatents,
    BaseDictEncoders_MultiLatents,
)

from ..base import BaseMultiVAE
from ..base.base_utils import poe, rsample_from_gaussian
from .mopoe_config import MoPoEConfig


class MoPoE(BaseMultiVAE):
    """Implementation for the Mixture of Product of experts model from
    'Generalized Multimodal ELBO' Sutter 2021 (https://arxiv.org/abs/2105.02470).

    This implementation is heavily based on the official one at
    https://github.com/thomassutter/MoPoE.

    Args:
        model_config (MoPoEConfig): Contains all the parameters for the model.
        encoders (dict): Contains the encoder for each modality. When using
            modalities' specific latent spaces, the encoders must be instances
            of ~multivae.models.nn.base_architectures.BaseMultilatentEncoder. Else,
            the encoders must be instances of ~pythae.models.nn.base_architectures.BaseEncoder.
            When None are provided, default MLP architectures are used.
        decoders (dict): Contains the decoder for each modality. Each decoder must be an
            instance of ~pythae.models.nn.base_architectures.BaseDecoder. When using modalities's
            specific latent spaces, the decoder takes as input the concatenation of both
            latent codes. When None are provided, default MLP architectures are used.

    """

    def __init__(
        self, model_config: MoPoEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.multiple_latent_spaces = model_config.modalities_specific_dim is not None
        self.model_name = "MoPoE"

        list_subsets = self.model_config.subsets

        if isinstance(list_subsets, dict):
            list_subsets = list(list_subsets.values())
        if list_subsets is None:
            list_subsets = self.all_subsets()
        self.set_subsets(list_subsets)

        if self.multiple_latent_spaces:
            self.style_dims = model_config.modalities_specific_dim
            # Set default architectures if not provided
            if encoders is None:
                encoders = BaseDictEncoders_MultiLatents(
                    input_dims=model_config.input_dims,
                    latent_dim=model_config.latent_dim,
                    modality_dims=model_config.modalities_specific_dim,
                )
                self.set_encoders(encoders)

            if decoders is None:
                decoders = BaseDictDecodersMultiLatents(
                    input_dims=model_config.input_dims,
                    latent_dim=model_config.latent_dim,
                    modality_dims=model_config.modalities_specific_dim,
                )
                self.set_decoders(decoders)

    def all_subsets(self):
        """Returns a list containing all possible subsets of the modalities.
        (But the empty one).
        """
        xs = list(self.encoders.keys())
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(
            combinations(xs, n) for n in range(len(xs) + 1)
        )
        return subsets_list

    def set_subsets(self, subsets_list):
        """Builds a dictionary of the subsets.
        The keys are the subset_names created by concatenating the modalities' names.
        The values are the list of modalities names.
        """
        subsets = dict()
        for mod_names in subsets_list:
            mods = []
            for mod_name in sorted(mod_names):
                if (mod_name not in self.encoders.keys()) and (mod_name != ""):
                    raise AttributeError(
                        f"The provided subsets list contains unknown modality name {mod_name}."
                        " that is not the encoders dictionary or inputs_dim dictionary."
                    )
                mods.append(mod_name)
            key = "_".join(sorted(mod_names))
            subsets[key] = mods
        self.subsets = subsets
        self.model_config.subsets = subsets
        return

    def calc_joint_divergence(
        self, mus: torch.Tensor, logvars: torch.Tensor, weights: torch.Tensor
    ):
        """Computes the KL divergence between the mixture of experts and the prior, by
        developping into the sum of the tractable KLs divergences of each expert.

        Args:
            mus (Tensor): The means of the experts. (n_subset,n_samples, latent_dim)
            logvars (Tensor): The logvars of the experts.(n_subset,n_samples, latent_dim)
            weights (Tensor): The weights of the experts.(n_subset,n_samples)


        Returns:
            Tensor, Tensor: The group divergence summed over modalities, A tensor containing the KL terms for each experts.
        """
        weights = weights.clone()

        num_mods = mus.shape[0]
        num_samples = mus.shape[1]
        klds = torch.zeros(num_mods, num_samples)

        device = mus.device
        klds = klds.to(device)
        weights = weights.to(device)
        for k in range(0, num_mods):
            kld_ind = -0.5 * (
                1 - logvars[k, :, :].exp() - mus[k, :, :].pow(2) + logvars[k, :, :]
            ).sum(-1)

            klds[k, :] = kld_ind

        group_div = (
            (weights * klds).sum(dim=0).mean()
        )  # sum over experts, mean over samples

        divs = dict()
        divs["joint_divergence"] = group_div
        return divs

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        # Compute latents parameters for all subsets
        latents = self.inference(inputs)
        results = dict()

        # Get the embeddings for shared latent space
        shared_embeddings = rsample_from_gaussian(
            latents["joint"][0], latents["joint"][1]
        )
        len_batch = shared_embeddings.shape[0]

        # Compute the divergence to the prior
        div = self.calc_joint_divergence(
            latents["mus"], latents["logvars"], latents["weights"]
        )
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        # Compute the reconstruction losses for each modality
        loss = 0
        kld = results["joint_divergence"]
        for m_key in self.encoders.keys():
            # reconstruct this modality from the shared embeddings representation

            if self.multiple_latent_spaces:
                try:
                    # sample from the modality specific latent space
                    style_mu = latents["modalities"][m_key].style_embedding
                    style_log_var = latents["modalities"][m_key].style_log_covariance
                    style_embeddings = rsample_from_gaussian(style_mu, style_log_var)
                    full_embedding = torch.cat(
                        [shared_embeddings, style_embeddings], dim=-1
                    )
                except:  # noqa
                    raise AttributeError(
                        " model_config.modality_specific_dims is not None, "
                        f"but encoder output for modality {m_key} doesn't have a "
                        "style_embedding attribute. "
                        "When using multiple latent spaces, the encoders' output"
                        "should be of the form : ModelOuput(embedding = ...,"
                        "style_embedding = ...,log_covariance = ..., style_log_covariance = ...)"
                    )
            else:
                full_embedding = shared_embeddings

            recon = self.decoders[m_key](full_embedding).reconstruction
            m_rec = (
                (
                    -self.recon_log_probs[m_key](recon, inputs.data[m_key])
                    * self.rescale_factors[m_key]
                )
                .view(recon.size(0), -1)
                .sum(-1)
            )

            # reconstruction loss
            if hasattr(inputs, "masks"):
                results["recon_" + m_key] = (m_rec * inputs.masks[m_key].float()).mean()
                # We still take the mean over the all batch even if some samples are missing
                # in order to give the same weights to all samples between batches
            else:
                results["recon_" + m_key] = m_rec.mean()

            loss += results["recon_" + m_key]

            # If using modality specific latent spaces, add modality specific klds
            if self.multiple_latent_spaces:
                style_mu = latents["modalities"][m_key].style_embedding
                style_log_var = latents["modalities"][m_key].style_log_covariance
                style_kld = -0.5 * (
                    1 - style_log_var.exp() - style_mu.pow(2) + style_log_var
                ).view(style_mu.size(0), -1).sum(-1)

                if hasattr(inputs, "masks"):
                    style_kld *= inputs.masks[m_key].float()

                kld += style_kld.mean() * self.model_config.beta_style

        loss = loss + self.model_config.beta * kld

        return ModelOutput(loss=loss, loss_sum=loss * len_batch, metrics=results)

    def modality_encode(
        self, inputs: Union[MultimodalBaseDataset, IncompleteDataset], **kwargs
    ):
        """Computes for each modality, the parameters mu and logvar of the
        unimodal posterior.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.

        Returns:
            dict: Containing for each modality the encoder output.
        """
        encoders_outputs = dict()
        for m, m_key in enumerate(self.encoders.keys()):
            input_modality = inputs.data[m_key]
            output = self.encoders[m_key](input_modality)
            encoders_outputs[m_key] = output

        return encoders_outputs

    def _poe_fusion(self, mus: torch.Tensor, logvars: torch.Tensor):
        # Following the original implementation : add the prior when we consider the
        # subset that contains all the modalities
        if mus.shape[0] == len(self.encoders.keys()):
            num_samples = mus[0].shape[0]
            device = mus.device
            mus = torch.cat(
                (mus, torch.zeros(1, num_samples, self.latent_dim).to(device)), dim=0
            )
            logvars = torch.cat(
                (logvars, torch.zeros(1, num_samples, self.latent_dim).to(device)),
                dim=0,
            )
        return poe(mus, logvars)

    def subset_mask(self, inputs: IncompleteDataset, subset: Union[list, tuple]):
        """Returns a filter of the samples available in ALL the modalities contained in subset."""
        filter = torch.tensor(
            True,
        ).to(inputs.masks[subset[0]].device)
        for mod in subset:
            filter = torch.logical_and(filter, inputs.masks[mod])

        return filter

    def inference(self, inputs: MultimodalBaseDataset, **kwargs):
        """Args:
            inputs (MultimodalBaseDataset): The data.

        Returns:
            dict: all the subset and joint posteriors parameters.
        """
        latents = dict()
        enc_mods = self.modality_encode(inputs)
        latents["modalities"] = enc_mods
        device = inputs.data[list(inputs.data.keys())[0]].device

        mus = torch.Tensor().to(device)
        logvars = torch.Tensor().to(device)
        distr_subsets = dict()
        availabilities = []

        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != "":
                mods = self.subsets[s_key]
                mus_subset = torch.Tensor().to(device)
                logvars_subset = torch.Tensor().to(device)

                if hasattr(inputs, "masks"):
                    filter = self.subset_mask(inputs, mods)
                    availabilities.append(filter)

                for m, mod in enumerate(mods):
                    mus_mod = enc_mods[mod].embedding
                    log_vars_mod = enc_mods[mod].log_covariance

                    mus_subset = torch.cat((mus_subset, mus_mod.unsqueeze(0)), dim=0)

                    logvars_subset = torch.cat(
                        (logvars_subset, log_vars_mod.unsqueeze(0)), dim=0
                    )

                # Case with only one sample : adapt the shape
                if len(mus_subset.shape) == 2:
                    mus_subset = mus_subset.unsqueeze(1)
                    logvars_subset = logvars_subset.unsqueeze(1)

                s_mu, s_logvar = self._poe_fusion(mus_subset, logvars_subset)

                distr_subsets[s_key] = [s_mu, s_logvar]

                # Add the subset posterior to be part of the mixture of experts
                mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)

        if hasattr(inputs, "masks"):
            # if we have an incomplete dataset, we need to randomly choose
            # from the mixture of available experts
            availabilities = torch.stack(availabilities, dim=0).float()
            if len(availabilities.shape) == 1:
                availabilities = availabilities.unsqueeze(1)
            availabilities /= torch.sum(availabilities, dim=0)  # (n_subset,n_samples)

            joint_mu, joint_logvar = self.random_mixture_component_selection(
                mus, logvars, availabilities
            )
            weights = availabilities
        else:
            weights = (1 / float(mus.shape[0])) * torch.ones(mus.shape[0]).to(device)
            joint_mu, joint_logvar = self.deterministic_mixture_component_selection(
                mus, logvars, weights
            )
            weights = (1 / float(mus.shape[0])) * torch.ones(
                mus.shape[0], mus.shape[1]
            ).to(device)

        latents["mus"] = mus
        latents["logvars"] = logvars
        latents["weights"] = weights
        latents["joint"] = [joint_mu, joint_logvar]
        latents["subsets"] = distr_subsets
        return latents

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        return_mean=False,
        **kwargs,
    ) -> ModelOutput:
        """Generate encodings conditioning on all modalities or a subset of modalities.
        We use the product of experts on the conditioning modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities
                names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.
            return_mean (bool) : if True, returns the mean of the posterior distribution (instead of a sample).

        Returns:
            ModelOutput instance with fields:
                z (torch.Tensor (n_data, N, latent_dim))
                one_latent_space (bool)
                modalities_z (Dict[str,torch.Tensor (n_data, N, latent_dim) ])

        """
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # Compute the str associated to the subset
        key = "_".join(sorted(cond_mod))

        latents_subsets = self.inference(inputs)

        # Take the product of experts on the subset
        mu, log_var = latents_subsets["subsets"][key]

        if return_mean and len(cond_mod) == self.n_modalities:
            # Aggregate the posterior
            mu = torch.stack(
                [latents_subsets["subsets"][k][0] for k in latents_subsets["subsets"]]
            ).mean(0)
        flatten = kwargs.pop("flatten", False)

        z = rsample_from_gaussian(mu, log_var, N, return_mean, flatten=flatten)

        if self.multiple_latent_spaces:
            modalities_z = {}
            for m in self.encoders:
                if m in cond_mod:
                    mu_style = latents_subsets["modalities"][m].style_embedding
                    log_var_style = latents_subsets["modalities"][
                        m
                    ].style_log_covariance
                else:
                    mu_style = torch.zeros((len(mu), self.style_dims[m])).to(mu.device)
                    log_var_style = torch.zeros((len(mu), self.style_dims[m])).to(
                        mu.device
                    )
                modalities_z[m] = rsample_from_gaussian(
                    mu_style, log_var_style, N, return_mean, flatten
                )

            return ModelOutput(z=z, one_latent_space=False, modalities_z=modalities_z)

        return ModelOutput(z=z, one_latent_space=True)

    def random_mixture_component_selection(self, mus, logvars, availabilities):
        """Randomly select a subset for each sample among the available subsets.

        Args:
            mus (tensor): (n_subset,n_samples,latent_dim) the means of subset posterior.
            logvars (tensor): (n_subset,n_samples,latent_dim) the log covariance of subset posterior.
            availabilities (tensor): (n_subset,n_samples) boolean tensor.
        """
        probs = availabilities.permute(1, 0)  # n_samples,n_subset
        choice = dist.OneHotCategorical(probs=probs).sample()

        mus_ = mus.permute(1, 0, 2)  # n_samples, n_subset,latent_dim
        logvars_ = logvars.permute(1, 0, 2)

        mus_ = mus_[choice.bool()]
        logvars_ = logvars_[choice.bool()]
        return mus_, logvars_

    def deterministic_mixture_component_selection(self, mus, logvars, w_modalities):
        """Associate a subset mu and log_covariance per sample in a balanced way, so that the proportion
        of samples per subset correspond to w_modalities.
        """
        num_components = mus.shape[0]  # number of components
        num_samples = mus.shape[1]

        idx_start = []
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0
            else:
                i_start = int(idx_end[k - 1])
            if k == w_modalities.shape[0] - 1:
                i_end = num_samples
            else:
                i_end = i_start + int(torch.floor(num_samples * w_modalities[k]))
            idx_start.append(i_start)
            idx_end.append(i_end)
        idx_end[-1] = num_samples
        mu_sel = torch.cat(
            [mus[k, idx_start[k] : idx_end[k], :] for k in range(w_modalities.shape[0])]
        )
        logvar_sel = torch.cat(
            [
                logvars[k, idx_start[k] : idx_end[k], :]
                for k in range(w_modalities.shape[0])
            ]
        )
        return [mu_sel, logvar_sel]

    @torch.no_grad()
    def compute_joint_nll(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Estimate the negative joint likelihood.
        In the original code, the product of experts is used as inference distribution
        for computing the nll instead of the MoPoe, but that is less coherent with the definition of the
        MoPoE definition as the joint posterior.

        Args:
            inputs (MultimodalBaseDataset) : a batch of samples.
            K (int) : the number of importance samples for the estimation. Default to 1000.
            batch_size_K (int) : Default to 100.

        Returns:
            The negative log-likelihood summed over the batch.


        """
        # Check that the dataset is complete
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The compute_joint_nll method is not yet implemented for incomplete datasets."
            )

        # Compute the parameters of each subset posterior
        infer = self.inference(inputs)
        mu, log_var = infer["joint"]  # a random subset is selected for each sample
        mus_subset = infer["mus"]
        log_vars_subset = infer["logvars"]

        # And sample from the posterior
        z_joint = rsample_from_gaussian(
            mu, log_var, N=K
        )  # shape K x n_data x latent_dim
        z_joint = z_joint.permute(1, 0, 2)
        n_data, _, _ = z_joint.shape

        # If using multiple latent spaces, sample from the private latent spaces as well
        private_params = {}
        private_zs = {}
        if self.multiple_latent_spaces:
            for mod in inputs.data:
                private_params[mod] = (
                    infer["modalities"][mod].style_embedding,
                    infer["modalities"][mod].style_log_covariance,
                )
                style_embeddings = rsample_from_gaussian(*private_params[mod], N=K)
                private_zs[mod] = style_embeddings.permute(
                    1, 0, 2
                )  # shape n_data x K x private dim

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            while start_idx < stop_idx:
                shared_latents = z_joint[i][start_idx:stop_idx]

                # Compute ln p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0
                lpz = 0
                lqz_xs = 0
                for mod in inputs.data:
                    if self.multiple_latent_spaces:
                        private_latents = private_zs[mod][i][start_idx:stop_idx]
                        full_embedding = torch.cat(
                            [shared_latents, private_latents], dim=-1
                        )
                    else:
                        full_embedding = shared_latents
                    decoder = self.decoders[mod]
                    recon = decoder(full_embedding)[
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

                    # If we are using modalities specific latent spaces compute the modalities priors and posteriors
                    if self.multiple_latent_spaces:
                        lpz += dist.Normal(0, 1).log_prob(private_latents).sum(dim=-1)
                        qz_x = dist.Normal(
                            private_params[mod][0][i],
                            torch.exp(0.5 * private_params[mod][1][i]),
                        )
                        lqz_xs += qz_x.log_prob(private_latents).sum(dim=-1)

                # Compute ln(p(z))
                prior = dist.Normal(0, 1)
                lpz += prior.log_prob(shared_latents).sum(dim=-1)

                # Compute shared posterior -ln(q(z|x,y) = -ln (1/S \sum q(z|x_s))
                qz_xs = [
                    dist.Normal(
                        mus_subset[j][i], torch.exp(0.5 * log_vars_subset[j][i])
                    )
                    for j in range(len(mus_subset))
                ]
                lqz_xs_tensor = torch.stack(
                    [q.log_prob(shared_latents).sum(-1) for q in qz_xs]
                )
                lqz_xs += torch.logsumexp(lqz_xs_tensor, dim=0) - np.log(
                    len(lqz_xs_tensor)
                )  # log_mean_exp

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xs, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll

    @torch.no_grad()
    def _compute_joint_nll_from_subset_encoding(
        self,
        subset,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Computes the joint negative log-likelihood using the PoE posterior as importance sampling distribution.
        The result is summed over the input batch.
        """
        # Check that the dataset is complete
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The compute_joint_nll method is not yet implemented for incomplete datasets."
            )

        subset_name = "_".join(sorted(subset))
        # Compute the parameters of the joint posterior
        infer = self.inference(inputs)
        mu, log_var = infer["subsets"][subset_name]

        # And sample from the posterior
        z_joint = rsample_from_gaussian(mu, log_var, K)  # shape K x n_data x latent_dim
        z_joint = z_joint.permute(1, 0, 2)
        n_data, _, _ = z_joint.shape

        # If using multiple latent spaces, sample from the private latent spaces as well
        private_params = {}
        private_zs = {}
        if self.multiple_latent_spaces:
            for mod in inputs.data:
                private_params[mod] = (
                    infer["modalities"][mod].style_embedding,
                    infer["modalities"][mod].style_log_covariance,
                )
                style_embeddings = rsample_from_gaussian(*private_params[mod], K)
                private_zs[mod] = style_embeddings.permute(
                    1, 0, 2
                )  # shape n_data x K x private dim

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)

            lnpxs = []
            while start_idx < stop_idx:
                shared_latents = z_joint[i][start_idx:stop_idx]

                # Compute ln p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0  # ln(p(x,y|z))
                lpz = 0  # ln(p(z)) prior
                lqz_xs = 0  # ln(q(z|X)) posterior
                for mod in inputs.data:
                    if self.multiple_latent_spaces:
                        private_latents = private_zs[mod][i][start_idx:stop_idx]
                        full_embedding = torch.cat(
                            [shared_latents, private_latents], dim=-1
                        )
                    else:
                        full_embedding = shared_latents

                    decoder = self.decoders[mod]
                    recon = decoder(full_embedding)[
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

                    # If we are using modalities specific latent spaces compute the modalities priors and posteriors
                    if self.multiple_latent_spaces:
                        lpz += dist.Normal(0, 1).log_prob(private_latents).sum(dim=-1)
                        qz_x = dist.Normal(
                            private_params[mod][0][i],
                            torch.exp(0.5 * private_params[mod][1][i]),
                        )
                        lqz_xs += qz_x.log_prob(private_latents).sum(dim=-1)

                # Compute ln(p(z))
                prior = dist.Normal(0, 1)
                lpz += prior.log_prob(shared_latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|x,y))
                qz_xy = dist.Normal(mu[i], torch.exp(0.5 * log_var[i]))
                lqz_xs += qz_xy.log_prob(shared_latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xs, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll

    def compute_joint_nll_paper(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Estimates the negative joint likelihood using the PoE posterior as importance sampling distribution.
        The result is summed over the input batch.

        This is the method used in the original paper implementation.
        """
        entire_subset = list(self.encoders.keys())
        return self._compute_joint_nll_from_subset_encoding(
            entire_subset, inputs, K, batch_size_K
        )
