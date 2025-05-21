import logging
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from torch.distributions import Laplace, Normal

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.utils import drop_unused_modalities
from multivae.models.nn.default_architectures import (
    BaseDictDecodersMultiLatents,
    BaseDictEncoders_MultiLatents,
)

from ..base import BaseMultiVAE
from .mmvaePlus_config import MMVAEPlusConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MMVAEPlus(BaseMultiVAE):
    """The MMVAE+ model.

    Args:
        model_config (MMVAEPlusConfig): An instance of MMVAEConfig in which any model's
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
        model_config: MMVAEPlusConfig,
        encoders: dict = None,
        decoders: dict = None,
    ):
        if model_config.modalities_specific_dim is None:
            raise AttributeError(
                "The modalities_specific_dim attribute must"
                " be provided in the model config."
            )

        super().__init__(model_config, encoders, decoders)

        if model_config.prior_and_posterior_dist == "laplace_with_softmax":
            self.post_dist = Laplace
            self.prior_dist = Laplace
        elif model_config.prior_and_posterior_dist == "normal":
            self.post_dist = Normal
            self.prior_dist = Normal
        elif model_config.prior_and_posterior_dist == "normal_with_softplus":
            self.post_dist = Normal
            self.prior_dist = Normal
        else:
            raise AttributeError(
                " The posterior_dist parameter must be "
                " either 'laplace_with_softmax','normal' or 'normal_with_softplus'. "
                f" {model_config.prior_and_posterior_dist} was provided."
            )

        # Set the priors for shared and private spaces.
        self.mean_priors = torch.nn.ParameterDict()
        self.logvars_priors = torch.nn.ParameterDict()
        self.beta = model_config.beta
        self.modalities_specific_dim = model_config.modalities_specific_dim
        self.reconstruction_option = model_config.reconstruction_option
        self.multiple_latent_spaces = True
        self.style_dims = {m: self.modalities_specific_dim for m in self.encoders}

        # Add the private and shared latents priors.

        # modality specific priors (referred to as r distributions in paper)
        for mod in list(self.encoders.keys()):
            self.mean_priors[mod] = torch.nn.Parameter(
                torch.zeros(1, model_config.modalities_specific_dim),
                requires_grad=False,
            )
            self.logvars_priors[mod] = torch.nn.Parameter(
                torch.zeros(1, model_config.modalities_specific_dim),
                requires_grad=model_config.learn_modality_prior,
            )

        # general prior (for the entire latent code) referred to as p in the paper
        self.mean_priors["shared"] = torch.nn.Parameter(
            torch.zeros(
                1, model_config.latent_dim + model_config.modalities_specific_dim
            ),
            requires_grad=False,
        )
        self.logvars_priors["shared"] = torch.nn.Parameter(
            torch.zeros(
                1, model_config.latent_dim + model_config.modalities_specific_dim
            ),
            requires_grad=model_config.learn_shared_prior,
        )

        self.model_name = "MMVAEPlus"
        self.objective = model_config.loss

    def _log_var_to_std(self, log_var):
        """For latent distributions parameters, transform the log covariance to the
        standard deviation of the distribution either applying softmax, softplus
        or simply torch.exp(0.5 * ...) depending on the model configuration.
        """
        if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
            return F.softmax(log_var, dim=-1) * log_var.size(-1) + 1e-6
        elif self.model_config.prior_and_posterior_dist == "normal_with_softplus":
            return F.softplus(log_var) + 1e-6
        else:
            return torch.exp(0.5 * log_var)

    def _compute_posteriors_and_embeddings(self, inputs, detach, **kwargs):
        # Drop unused modalities
        inputs = drop_unused_modalities(inputs)

        # First compute all the encodings for all modalities
        embeddings = {}
        posteriors = {m: {} for m in inputs.data}
        reconstructions = {}

        k_iwae = kwargs.pop("K", self.model_config.K)

        for cond_mod in inputs.data:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance
            mu_style = output.style_embedding
            log_var_style = output.style_log_covariance

            sigma = self._log_var_to_std(log_var)
            sigma_style = self._log_var_to_std(log_var_style)

            # Shared latent variable
            qu_x = self.post_dist(mu, sigma)
            u_x = qu_x.rsample([k_iwae])

            # Private latent variable
            qw_x = self.post_dist(mu_style, sigma_style)
            w_x = qw_x.rsample([k_iwae])

            # The DREG loss uses detached parameters in the loss computation afterwards.
            if detach:
                qu_x = self.post_dist(mu.clone().detach(), sigma.clone().detach())
                qw_x = self.post_dist(
                    mu_style.clone().detach(), sigma_style.clone().detach()
                )

            # Then compute all the cross-modal reconstructions
            reconstructions[cond_mod] = {}

            for recon_mod in inputs.data:
                # Self-reconstruction
                if recon_mod == cond_mod:
                    z_x = torch.cat([u_x, w_x], dim=-1)

                # Cross modal reconstruction
                else:
                    # only keep the shared latent and generate private from prior

                    mu_prior_mod = torch.cat(
                        [self.mean_priors[recon_mod]] * len(mu), axis=0
                    )
                    sigma_prior_mod = torch.cat(
                        [self._log_var_to_std(self.logvars_priors[recon_mod])]
                        * len(mu),
                        axis=0,
                    )

                    w = self.prior_dist(
                        mu_prior_mod,
                        sigma_prior_mod,
                    ).rsample([k_iwae])

                    z_x = torch.cat([u_x, w], dim=-1)
                # Decode

                z = z_x.reshape(-1, z_x.shape[-1])
                recon = self.decoders[recon_mod](z)["reconstruction"]
                recon = recon.reshape((*z_x.shape[:-1], *recon.shape[1:]))

                reconstructions[cond_mod][recon_mod] = recon

            posteriors[cond_mod] = {"u": qu_x, "w": qw_x}
            embeddings[cond_mod] = {"u": u_x, "w": w_x}

        return embeddings, posteriors, reconstructions

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """Compute loss and metrics."""
        if self.objective == "dreg_looser":
            # The DreG estimation uses detached posteriors
            embeddings, posteriors, reconstructions = (
                self._compute_posteriors_and_embeddings(inputs, detach=True)
            )
            return self._dreg_looser(posteriors, embeddings, reconstructions, inputs)

        if self.objective == "iwae_looser":
            embeddings, posteriors, reconstructions = (
                self._compute_posteriors_and_embeddings(inputs, detach=False)
            )
            return self._iwae_looser(posteriors, embeddings, reconstructions, inputs)
        raise NotImplementedError

    @property
    def pz_params(self):
        """From the prior mean and log_covariance, return the mean and standard
        deviation, either applying softmax or not depending on the choice of prior
        distribution.

        Returns:
            tuple: mean, std
        """
        mean = self.mean_priors["shared"]
        log_var = self.logvars_priors["shared"]
        std = self._log_var_to_std(log_var)
        return mean, std

    def _compute_k_lws(self, posteriors, embeddings, reconstructions, inputs):
        """Compute the individual likelihoods without any aggregation on k_iwae
        or the batch.
        """
        if hasattr(inputs, "masks"):
            # Compute the number of available modalities per sample
            n_mods_sample = torch.sum(
                torch.stack(tuple(inputs.masks.values())).int(), dim=0
            )
        else:
            n_mods_sample = torch.tensor([self.n_modalities])

        lws = {}
        for mod in embeddings:
            u = embeddings[mod]["u"]  # (K, n_batch, latent_dim)
            w = embeddings[mod]["w"]  # (K, n_batch, latent_dim)
            n_mods_sample = n_mods_sample.to(u.device)

            ### Compute log p(z)
            z = torch.cat([u, w], dim=-1)
            lpz = self.prior_dist(*self.pz_params).log_prob(z).sum(-1)

            ### Compute log q(u|X) where u is the shared latent
            # Get all the individual log q(u|x_i) for all modalities
            if hasattr(inputs, "masks"):
                qu_x = []
                for m in posteriors:
                    qu = posteriors[m]["u"].log_prob(u).sum(-1)
                    # for unavailable modalities, set the log prob to -infinity so that it accounts for 0
                    # in the log_sum_exp.
                    qu[torch.stack([inputs.masks[m] == False] * len(u))] = -torch.inf
                    qu_x.append(qu)
                lqu_x = torch.stack(qu_x)  # n_modalities,K,nbatch
            else:
                lqu_x = torch.stack(
                    [posteriors[m]["u"].log_prob(u).sum(-1) for m in posteriors]
                )  # n_modalities,K,nbatch

            # Compute the mixture of experts
            lqu_x = torch.logsumexp(lqu_x, dim=0) - torch.log(
                n_mods_sample
            )  # log_mean_exp

            ### Compute log q(w |x_m)
            lqw_x = posteriors[mod]["w"].log_prob(w).sum(-1)

            ### Compute log p(X|u,w) for all modalities
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

            ### Compute the entire likelihood
            lw = lpx_z + self.beta * (lpz - lqu_x - lqw_x)

            if hasattr(inputs, "masks"):
                # cancel unavailable modalities
                lw *= inputs.masks[mod].float()

            lws[mod] = lw

        return lws, n_mods_sample

    def _dreg_looser(self, posteriors, embeddings, reconstructions, inputs):
        """The DreG estimation for IWAE. losses components in lws needs to have been computed on
        **detached** posteriors.
        """
        lws, n_mods_sample = self._compute_k_lws(
            posteriors, embeddings, reconstructions, inputs
        )

        ### Compute the wk for each modality
        wk = {}
        with torch.no_grad():  # The wk are constants
            for m, lw in lws.items():
                wk[m] = (
                    lw - torch.logsumexp(lw, 0, keepdim=True)
                ).exp()  # K, batch_size

        ### Compute the loss
        lws = torch.stack(
            [lws[mod] * wk[mod] for mod in lws], dim=0
        )  # n_modalities, K, batch_size
        lws = lws.sum(1)  # Sum over the k_iwae samples

        ### Take the mean over the modalities (outside the log)
        lws = lws.sum(0) / n_mods_sample

        # The gradient with respect to \phi is multiplied one more time by wk
        # To achieve that, we register a hook on the latent variables u and w
        for mod in embeddings:
            embeddings[mod]["w"].register_hook(
                lambda grad, w=wk[mod]: w.unsqueeze(-1) * grad
            )
            embeddings[mod]["u"].register_hook(
                lambda grad, w=wk[mod]: w.unsqueeze(-1) * grad
            )

        ### Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics=dict())

    def _iwae_looser(self, posteriors, embeddings, reconstructions, inputs):
        """The IWAE loss but with the sum outside of the loss for increased stability.
        (following Shi et al 2019).

        """
        # Get all individual likelihoods
        lws, n_mods_sample = self._compute_k_lws(
            posteriors, embeddings, reconstructions, inputs
        )
        lws = torch.stack(list(lws.values()), dim=0)  # (n_modalities, K, n_batch)

        # Take log_mean_exp on the k_iwae samples to obtain the k-sampled estimate
        lws = torch.logsumexp(lws, dim=1) - math.log(
            lws.size(1)
        )  # n_modalities, n_batch

        # Take the mean on modalities
        lws = lws.sum(0) / n_mods_sample

        # Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics=dict())

    def encode(
        self,
        inputs: MultimodalBaseDataset,
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
            ModelOutput : contains fields
                'z' (torch.Tensor (n_data, N, latent_dim))
                'one_latent_space' (bool) = False
                'modalities_z' (Dict[str,torch.Tensor (n_data, N, latent_dim) ])
        """
        # Look up the batchsize
        batch_size = len(list(inputs.data.values())[0])

        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod
        if all(s in self.encoders.keys() for s in cond_mod):
            # For the conditioning modalities we compute all the embeddings
            encoders_outputs = {m: self.encoders[m](inputs.data[m]) for m in cond_mod}

            if return_mean:
                list_mean = [o.embedding for o in encoders_outputs.values()]
                embedding = torch.mean(torch.stack(list_mean), dim=0)
                z = torch.stack([embedding] * N) if N > 1 else embedding
            else:
                # Choose one of the conditioning modalities at random to sample the shared information.
                random_mod = np.random.choice(cond_mod)

                # Sample the shared latent code
                mu = encoders_outputs[random_mod].embedding
                sigma = self._log_var_to_std(
                    encoders_outputs[random_mod].log_covariance
                )

                sample_shape = torch.Size([]) if N == 1 else torch.Size([N])
                z = self.post_dist(mu, sigma).rsample(sample_shape)

            flatten = kwargs.pop("flatten", False)
            if flatten:
                z = z.reshape(-1, self.latent_dim)

            # Modality specific encodings : given by encoders for conditioning modalities
            # Sampling from the priors for the rest of the modalities.

            style_z = {}

            for m in self.encoders:
                if m not in cond_mod:
                    # Sample from priors parameters.
                    if self.reconstruction_option == "single_prior":
                        mu_m = self.mean_priors[m]
                        logvar_m = self.logvars_priors[m]

                    if self.reconstruction_option == "joint_prior":
                        mu_m = self.mean_priors["shared"][:, self.latent_dim :]
                        logvar_m = self.logvars_priors["shared"][:, self.latent_dim :]

                    mu_m = torch.cat([mu_m] * batch_size, dim=0)
                    logvar_m = torch.cat([logvar_m] * batch_size, dim=0)

                else:
                    # Sample from posteriors parameters
                    mu_m = encoders_outputs[m].style_embedding
                    logvar_m = encoders_outputs[m].style_log_covariance

                sigma_m = self._log_var_to_std(logvar_m)

                if return_mean:
                    style_z[m] = torch.stack([mu_m] * N) if N > 1 else mu_m
                else:
                    style_z[m] = self.post_dist(mu_m, sigma_m).rsample(sample_shape)
                if flatten:
                    style_z[m] = style_z[m].reshape(-1, self.modalities_specific_dim)

            return ModelOutput(z=z, one_latent_space=False, modalities_z=style_z)

    def generate_from_prior(self, n_samples, **kwargs):
        sample_shape = [n_samples] if n_samples > 1 else []
        z = self.prior_dist(*self.pz_params).rsample(sample_shape).to(self.device)
        return ModelOutput(z=z.squeeze(), one_latent_space=True)

    def default_encoders(self, model_config) -> nn.ModuleDict:
        return BaseDictEncoders_MultiLatents(
            input_dims=model_config.input_dims,
            latent_dim=model_config.latent_dim,
            modality_dims={
                m: model_config.modalities_specific_dim
                for m in self.model_config.input_dims
            },
        )

    def default_decoders(self, model_config) -> nn.ModuleDict:
        return BaseDictDecodersMultiLatents(
            input_dims=model_config.input_dims,
            latent_dim=model_config.latent_dim,
            modality_dims={
                m: model_config.modalities_specific_dim for m in model_config.input_dims
            },
        )

    @torch.no_grad()
    def compute_joint_nll(self, inputs, K=1000, batch_size_K=100):
        """Estimate the negative joint likelihood.

        Args:
            inputs (MultimodalBaseDataset) : a batch of samples.
            K (int) : the number of importance samples for the estimation. Default to 1000.
            batch_size_K (int) : Default to 100.

        Returns:
            The negative log-likelihood summed over the batch.
        """
        # Check that the dataset is not incomplete
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The compute_joint_nll method is not yet implemented for incomplete datasets."
            )

        n_data = len(inputs.data.popitem()[1])  # number of samples in the dataset

        ll = 0

        # Set the rescale factors and beta to 1 for the computation of the likelihood
        rescale_factors, self.rescale_factors = (
            self.rescale_factors.copy(),
            {m: 1 for m in self.rescale_factors},
        )
        beta, self.beta = self.model_config.beta, 1

        for i in range(n_data):
            inputs_i = MultimodalBaseDataset(
                data={m: inputs.data[m][i].unsqueeze(0) for m in inputs.data}
            )
            k_iwae = K // self.n_modalities  # number of samples per modality
            embeddings, posteriors, reconstructions = (
                self._compute_posteriors_and_embeddings(
                    inputs_i, detach=False, K=k_iwae
                )
            )

            lws, _ = self._compute_k_lws(
                posteriors, embeddings, reconstructions, inputs_i
            )

            # aggregate by taking the logsumexp on all lws element
            lws = torch.cat(list(lws.values()), dim=0)  # n_modalities*K, n_batch

            # Take log_mean_exp on all samples
            ll += torch.logsumexp(lws, dim=0) - math.log(lws.size(0))  # n_batch

        # revert changes made on rescale factors and beta
        self.rescale_factors = rescale_factors
        self.beta = beta

        return -ll.sum()
