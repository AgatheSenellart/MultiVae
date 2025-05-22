import math
from typing import Dict, Union

import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder
from torch import nn

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset
from multivae.models.nn.default_architectures import (
    BaseDictDecodersMultiLatents,
    BaseDictEncoders_MultiLatents,
)

from ..base import BaseMultiVAE
from ..base.base_utils import kl_divergence, rsample_from_gaussian, stable_poe
from ..nn.base_architectures import BaseMultilatentEncoder
from .dmvae_config import DMVAEConfig


class DMVAE(BaseMultiVAE):
    """The DMVAE model from the paper
    'Private-Shared Disentangled Multimodal VAE for Learning of Latent Representations'.

    Mihee Lee, Vladimir Pavlovic

    Args:
        model_config (DMVAEConfig): An instance of DMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~multivae.models.nn.base_architectures.BaseMultilatentEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Multivae's BaseMultilatentEncoder since this model uses multiple latent spaces. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.
    """

    def __init__(
        self,
        model_config: DMVAEConfig,
        encoders: Union[Dict[str, BaseMultilatentEncoder], None] = None,
        decoders: Union[Dict[str, BaseDecoder], None] = None,
    ):
        super().__init__(model_config, encoders, decoders)

        self.beta = model_config.beta
        self.model_name = "DMVAE"
        self._set_private_betas(model_config.modalities_specific_betas)
        self._set_modalities_specific_dim(model_config)
        self.multiple_latent_spaces = True

    def _set_modalities_specific_dim(self, model_config):
        if model_config.modalities_specific_dim is None:
            self.style_dims = {m: 1.0 for m in self.encoders}
        else:
            if model_config.modalities_specific_dim.keys() != self.encoders.keys():
                raise AttributeError(
                    "The keys in modalities_specific_dim doesn't match ",
                    "the keys in the encoders or input_dims",
                )

            else:
                self.style_dims = model_config.modalities_specific_dim
        return

    def _set_private_betas(self, beta_dict):
        if beta_dict is None:
            self.private_betas = {mod: 1.0 for mod in self.encoders}
        else:
            if not self.encoders.keys() == beta_dict.keys():
                raise AttributeError(
                    "The modality_specific_betas doesn't have the same "
                    "keys (modalities) as the provided encoders dict."
                )
            self.private_betas = beta_dict

    def default_encoders(self, model_config) -> nn.ModuleDict:
        return BaseDictEncoders_MultiLatents(
            input_dims=model_config.input_dims,
            latent_dim=model_config.latent_dim,
            modality_dims=model_config.modalities_specific_dim,
        )

    def default_decoders(self, model_config) -> nn.ModuleDict:
        return BaseDictDecodersMultiLatents(
            input_dims=model_config.input_dims,
            latent_dim=model_config.latent_dim,
            modality_dims=model_config.modalities_specific_dim,
        )

    def _infer_latent_parameters(self, inputs, subset=None):
        """Compute the latent parameters for the shared and private latent spaces,
        taking the product-of-experts on the subset.
        """
        # if no subset is provided, use all available modalities
        if subset is None:
            subset = list(inputs.data.keys())

        # First compute all embeddings
        private_params = {}
        shared_params = {}

        for mod in subset:
            output_mod = self.encoders[mod](inputs.data[mod])
            private_params[mod] = (
                output_mod.style_embedding,
                output_mod.style_log_covariance,
            )
            if len(output_mod.style_embedding.shape) == 1:
                private_params[mod] = (
                    output_mod.style_embedding.unsqueeze(0),
                    output_mod.style_log_covariance.unsqueeze(0),
                )

            shared_params[mod] = (output_mod.embedding, output_mod.log_covariance)
            if len(output_mod.embedding.shape) == 1:
                shared_params[mod] = (
                    output_mod.embedding.unsqueeze(0),
                    output_mod.log_covariance.unsqueeze(0),
                )

        # Compute the PoE for the shared variable
        list_mu = [shared_params[mod][0] for mod in shared_params]
        list_lvs = []

        # For unavailable modalities, set the variance to infinity so that it doesn't count in the PoE
        for mod in subset:
            log_var_mod = shared_params[mod][1].clone()
            if hasattr(inputs, "masks"):
                log_var_mod[(1 - inputs.masks[mod].int()).bool().flatten()] = torch.inf
            list_lvs.append(log_var_mod)

        # Add N(0,I) prior to the product of experts
        list_mu.append(torch.zeros_like(list_mu[0]))
        list_lvs.append(torch.zeros_like(list_lvs[0]))

        joint_mu, joint_lv = stable_poe(torch.stack(list_mu), torch.stack(list_lvs))
        return joint_mu, joint_lv, shared_params, private_params

    def forward(
        self, inputs: Union[MultimodalBaseDataset, IncompleteDataset], **kwargs
    ):
        """The main function of the model that computes the loss and some monitoring metrics.
        One of the advantages of DMVAE is that we can train with incomplete data.

        Args:
            inputs (MultimodalBaseDataset): The data. It can be an instance of IncompleteDataset
                which contains a field masks for weakly supervised learning.
                masks is a dictionary indicating which datasamples are missing
                in each of the modalities.
                For each modality, a boolean tensor indicates which samples are available. (The non
                available samples are assumed to be replaced with zero values in the multimodal dataset entry.)
        """
        (
            joint_mu,
            joint_lv,
            shared_params,
            private_params,
        ) = self._infer_latent_parameters(inputs)

        metrics = dict()
        # Compute the joint elbo
        joint_elbo = self._compute_elbo(joint_mu, joint_lv, private_params, inputs)
        loss = joint_elbo
        metrics["joint"] = joint_elbo.mean()

        # Compute crossmodal elbos
        for k, params in shared_params.items():
            mod_elbo = self._compute_elbo(params[0], params[1], private_params, inputs)

            if hasattr(inputs, "masks"):
                mod_elbo = inputs.masks[k] * mod_elbo

            loss += mod_elbo

            metrics[k] = mod_elbo.mean()

        return ModelOutput(loss=loss.mean(), metrics=metrics)

    def _compute_elbo(self, q_mu, q_lv, private_params, inputs):
        shared_z = rsample_from_gaussian(q_mu, q_lv)

        # Compute reconstructions
        recon_loss = 0
        for mod in self.encoders:
            # Sample the modality specific
            mu, logvar = private_params[mod]
            z_mod = rsample_from_gaussian(mu, logvar)

            z = torch.cat([shared_z, z_mod], dim=1)

            recon_mod = self.decoders[mod](z).reconstruction

            recon_mod = (
                self.recon_log_probs[mod](recon_mod, inputs.data[mod])
                * self.rescale_factors[mod]
            )
            recon_mod = recon_mod.reshape(recon_mod.size(0), -1).sum(-1)

            if hasattr(inputs, "masks"):
                # filter unavailable modalities in the reconstruction loss
                recon_mod = inputs.masks[mod].float() * recon_mod

            recon_loss += recon_mod

        # Compute KL divergence for shared variable
        shared_kl = kl_divergence(
            q_mu, q_lv, torch.zeros_like(q_mu), torch.zeros_like(q_lv)
        )

        kl = shared_kl * self.beta
        # Add the modality specific kls
        for mod in self.encoders:
            mu, lv = private_params[mod]
            kl_mod = kl_divergence(mu, lv, torch.zeros_like(mu), torch.zeros_like(lv))

            kl_mod = kl_mod.reshape(kl_mod.size(0), -1).sum(-1)

            if hasattr(inputs, "masks"):
                kl_mod = inputs.masks[mod].float() * kl_mod

            kl += kl_mod * self.private_betas[mod]

        neg_elbo = -recon_loss + kl

        return neg_elbo

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
            ModelOutput : Contains fields
                'z' (torch.Tensor (N, n_data, latent_dim))
                'one_latent_space' (bool) = False
                'modalities_z' (dict[str,torch.Tensor (N, n_data,mod_latent_dim)])
        """
        # Call super to perform some checks and preprocess the cond_mod argument
        # you obtain a list of the modalities' names to condition on
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # Compute the shared latent variable conditioning on input modalities
        sub_mu, sub_logvar, _, private_params = self._infer_latent_parameters(
            inputs, cond_mod
        )
        flatten = kwargs.pop("flatten", False)

        z = rsample_from_gaussian(
            sub_mu, sub_logvar, N=N, return_mean=return_mean, flatten=flatten
        )

        modalities_z = {}
        for mod in self.encoders:
            if mod in cond_mod:
                mod_mu, mod_lv = private_params[mod]
            else:
                mod_mu = torch.zeros((sub_mu.shape[0], self.style_dims[mod])).to(
                    sub_mu.device
                )
                mod_lv = torch.zeros_like(mod_mu).to(sub_logvar.device)

            modalities_z[mod] = rsample_from_gaussian(
                mod_mu, mod_lv, N=N, return_mean=return_mean, flatten=flatten
            )

        return ModelOutput(z=z, one_latent_space=False, modalities_z=modalities_z)

    def generate_from_prior(self, n_samples, **kwargs):
        """Generates latent variables from the prior for the shared latent spaces and
        for each modality specific latent space.

        Args:
            n_samples
        """
        device = self.device if self.device is not None else "cpu"

        # Generate shared latent variable
        shared_latent_shape = (
            [n_samples, self.latent_dim] if n_samples > 1 else [self.latent_dim]
        )
        z_shared = dist.Normal(0, 1).rsample(shared_latent_shape).to(device)

        # Generate modalities specific variables
        modalities_z = {}

        for k, dim in self.style_dims.items():
            shape = [n_samples, dim] if n_samples > 1 else [dim]
            modalities_z[k] = dist.Normal(0, 1).rsample(shape).to(device)

        return ModelOutput(
            z=z_shared, one_latent_space=False, modalities_z=modalities_z
        )

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

        # Compute the parameters of the joint posterior for the shared latent space
        # Compute the shared latent variable conditioning on input modalities
        mu, log_var, _, private_params = self._infer_latent_parameters(inputs)

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)

        # Sample K latents from the shared joint posterior
        z_joint = qz_xy.rsample([K]).permute(
            1, 0, 2
        )  # shape :  n_data x K x latent_dim
        n_data, _, _ = z_joint.shape

        # iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        ln_prior, ln_posterior = 0, 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            # iterate over the mini-batch for the K samples
            while start_idx < stop_idx:
                shared_latents = z_joint[i][start_idx:stop_idx]
                # Compute ln p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0
                for mod in inputs.data:
                    # Sample from the modality specific latent space
                    mu_private, logvar_private = private_params[mod]
                    mu_private, sigma_private = (
                        mu_private[i],
                        torch.exp(0.5 * logvar_private[i]),
                    )
                    private_latents = dist.Normal(mu_private, sigma_private).rsample(
                        [len(shared_latents)]
                    )

                    latents = torch.cat([shared_latents, private_latents], dim=-1)

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

                    # Compute ln(p(z_private))
                    ln_prior += dist.Normal(0, 1).log_prob(private_latents).sum(dim=-1)

                    # Compute ln(q(z_private|x))
                    ln_posterior += (
                        dist.Normal(mu_private, sigma_private)
                        .log_prob(private_latents)
                        .sum(-1)
                    )

                # Compute ln(p(z_shared))
                prior = dist.Normal(0, 1)
                ln_prior += prior.log_prob(shared_latents).sum(dim=-1)

                # Compute posteriors ln(q(z_shared|x,y))
                qz_xy = dist.Normal(mu[i], sigma[i])
                ln_posterior += qz_xy.log_prob(shared_latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + ln_prior - ln_posterior, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - math.log(K)

        return -ll
