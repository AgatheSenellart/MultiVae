from itertools import combinations
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from numpy.random import choice
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from ..base.base_utils import rsample_from_gaussian, stable_poe
from .mvae_config import MVAEConfig


class MVAE(BaseMultiVAE):
    """The Multi-modal VAE model.

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
        self._set_subsets()
        self.warmup = model_config.warmup
        self.start_keep_best_epoch = model_config.warmup + 1
        self.beta = model_config.beta
        self.model_name = "MVAE"

    def _set_subsets(self):
        self.subsets = []
        for i in range(2, self.n_modalities):
            self.subsets += combinations(list(self.encoders.keys()), r=i)

    def compute_mu_log_var_subset(self, inputs: MultimodalBaseDataset, subset: list):
        """Computes the parameters of the posterior when conditioning on
        the modalities contained in subset.
        """
        mus_sub = []
        log_vars_sub = []
        for mod in self.encoders:
            if mod in subset:
                output_mod = self.encoders[mod](inputs.data[mod])
                mu_mod, log_var_mod = output_mod.embedding, output_mod.log_covariance

                # set variance to inf for missing modalities so that they are not taken into account
                # in the product of experts

                if hasattr(inputs, "masks"):
                    log_var_mod[(1 - inputs.masks[mod].int()).bool().flatten()] = (
                        torch.inf
                    )

                mus_sub.append(mu_mod)
                log_vars_sub.append(log_var_mod)

        # Add the prior to the product of experts
        mus_sub.append(torch.zeros_like(mus_sub[0]))
        log_vars_sub.append(torch.zeros_like(log_vars_sub[0]))
        # Compute the Product of Experts
        sub_mu, sub_logvar = stable_poe(torch.stack(mus_sub), torch.stack(log_vars_sub))
        return sub_mu, sub_logvar

    def _compute_elbo_subset(
        self, inputs: MultimodalBaseDataset, subset: list, beta: float
    ):
        sub_mu, sub_logvar = self.compute_mu_log_var_subset(inputs, subset)
        z = rsample_from_gaussian(sub_mu, sub_logvar)
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

        return (
            elbo_sub / len(sub_mu),
            KLD / len(sub_mu),
            recon / len(sub_mu),
            len(sub_mu),
        )

    def _filter_inputs_with_masks(
        self, inputs: IncompleteDataset, subset: Union[list, tuple]
    ):
        """Returns a filtered dataset containing only the samples that are available
        in at least one of the modalities contained in subset.
        """
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
        # The annealing factor is updated each batch, so we need to know the idx of the batch in the epoch
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
            if self.k > 0 and self.training:
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
                (
                    subset_elbo,
                    subset_kld,
                    subset_recon,
                    len_batch,
                ) = self._compute_elbo_subset(filtered_inputs, s, beta)

                # update the metrics for monitoring
                metrics["_".join(sorted(s))] = subset_elbo
                metrics["beta"] = beta
                metrics["kld" + "_".join(sorted(s))] = subset_kld
                metrics["recon" + "_".join(sorted(s))] = subset_recon
            else:
                subset_elbo = subset_kld = subset_recon = torch.tensor(
                    0.0, requires_grad=True
                )
                len_batch = 0.0
            total_loss += subset_elbo

        return ModelOutput(
            loss=total_loss, loss_sum=total_loss * len_batch, metrics=metrics
        )

    def encode(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
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
            ModelOutput : contains `z` (torch.Tensor (n_data, N, latent_dim)), `one_latent_space` (bool) = True

        """
        # Call super to perform some checks and preprocess the cond_mod argument
        # you obtain a list of the modalities' names to condition on
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # Compute the latent variable conditioning on input modalities
        sub_mu, sub_logvar = self.compute_mu_log_var_subset(inputs, cond_mod)
        flatten = kwargs.pop("flatten", False)
        z = rsample_from_gaussian(
            sub_mu, sub_logvar, N=N, return_mean=return_mean, flatten=flatten
        )

        return ModelOutput(z=z, one_latent_space=True)

    @torch.no_grad()
    def compute_joint_nll(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        """Estimate the negative joint likelihood.

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

        # Compute the parameters of the joint posterior
        mu, log_var = self.compute_mu_log_var_subset(inputs, list(self.encoders.keys()))
        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)

        # Sample K latents from the joint posterior
        z_joint = qz_xy.rsample([K]).permute(
            1, 0, 2
        )  # shape :  n_data x K x latent_dim
        n_data, _, _ = z_joint.shape

        # iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            # iterate over the mini-batch for the K samples
            while start_idx < stop_idx:
                latents = z_joint[i][start_idx:stop_idx]

                # Compute ln p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0
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
                qz_xy = dist.Normal(mu[i], sigma[i])
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
