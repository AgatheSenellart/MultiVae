import logging
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from torch.distributions import Laplace, Normal

from multivae.data.datasets.base import MultimodalBaseDataset
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

    """
    The MMVAE+ model.

    Args:
        model_config (MMVAEPlusConfig): An instance of MMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

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

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        # TODO : maybe implement a minibatch strategy for stashing the gradients before
        # backpropagation when using a large number k.
        # Also, I've only implemented the dreg_looser loss but it may be nice to offer other options.

        # First compute all the encodings for all modalities
        embeddings = {}
        qu_xs = {}
        qw_xs = {}
        qu_xs_detach = {}
        qw_xs_detach = {}

        reconstructions = {}

        compute_loss = kwargs.pop("compute_loss", True)
        detailed_output = kwargs.pop("detailed_output", False)
        K = kwargs.pop("K", self.K)

        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance
            mu_style = output.style_embedding
            log_var_style = output.style_log_covariance

            sigma = self.log_var_to_std(log_var)
            sigma_style = self.log_var_to_std(log_var_style)

            # Shared latent variable
            qu_x = self.post_dist(mu, sigma)
            u_x = qu_x.rsample([K])

            # Private latent variable
            qw_x = self.post_dist(mu_style, sigma_style)
            w_x = qw_x.rsample([K])

            # The DREG loss uses detached parameters in the loss computation afterwards.
            qu_x_detach = self.post_dist(mu.detach(), sigma.detach())
            qw_x_detach = self.post_dist(mu_style.detach(), sigma_style.detach())

            # Then compute all the cross-modal reconstructions
            reconstructions[cond_mod] = {}

            for recon_mod in self.decoders:
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
                        [self.log_var_to_std(self.logvars_priors[recon_mod])] * len(mu),
                        axis=0,
                    )

                    w = self.prior_dist(
                        mu_prior_mod,
                        sigma_prior_mod,
                    ).rsample([K])
                    # print(w.shape)
                    # print(u_x.shape,w.shape)
                    z_x = torch.cat([u_x, w], dim=-1)
                # Decode
                # print(z_x.shape)
                decoder = self.decoders[recon_mod]
                recon = decoder(z_x)["reconstruction"]

                reconstructions[cond_mod][recon_mod] = recon

            qu_xs[cond_mod] = qu_x
            qu_xs_detach[cond_mod] = qu_x_detach
            qw_xs[cond_mod] = qw_x
            qw_xs_detach[cond_mod] = qw_x_detach
            embeddings[cond_mod] = dict(u=u_x, w=w_x)

        # Compute DREG loss
        if compute_loss:
            # loss_output = self.dreg_looser(
            #     qu_xs_detach, qw_xs_detach, embeddings, reconstructions, inputs
            # )
            loss_output = self.dreg_looser(
                qu_xs, qw_xs, embeddings, reconstructions, inputs
            )

        else:
            loss_output = ModelOutput()
        if detailed_output:
            loss_output["zss"] = embeddings
            loss_output["recon"] = reconstructions

        return loss_output

    @property
    def pz_params(self):
        """From the prior mean and log_covariance, return the mean and standard
        deviation, either applying softmax or not depending on the choice of prior
        distribution.

        Returns:
            tuple: mean, std
        """
        mean = self.mean_priors["shared"]
        if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
            std = F.softmax(
                self.logvars_priors["shared"], dim=-1
            ) * self.logvars_priors["shared"].size(-1)
        else:
            std = torch.exp(0.5 * self.logvars_priors["shared"])
        return mean, std

    def dreg_looser(self, qu_xs, qw_xs, embeddings, reconstructions, inputs):
        """
        The objective function used in the original implementation.
        it is said to use the Dreg estimator but I am not sure it is done correctly since it doesn't
        detach the parameters of the posteriors.
        """

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
            # Log-probabilities to the prior are the same as in MMVAE
            u = embeddings[mod]["u"]  # (K, n_batch, latent_dim)
            w = embeddings[mod]["w"]  # (K, n_batch, latent_dim)
            n_mods_sample = n_mods_sample.to(u.device)
            prior = self.prior_dist(*self.pz_params)
            z = torch.cat([u, w], dim=-1)
            lpz = prior.log_prob(z).sum(-1)

            # For the shared latent variable it is the same
            if hasattr(inputs, "masks"):
                lqu_x = torch.stack(
                    [
                        qu_xs[m].log_prob(u).sum(-1) * inputs.masks[m].float()
                        for m in qu_xs
                    ]
                )  # n_modalities,K,nbatch
            else:
                lqu_x = torch.stack(
                    [qu_xs[m].log_prob(u).sum(-1) for m in qu_xs]
                )  # n_modalities,K,nbatch

            lqu_x = torch.logsumexp(lqu_x, dim=0) - torch.log(
                n_mods_sample
            )  # log_mean_exp

            # Then we have to add the modality specific posterior
            lqw_x = qw_xs[mod].log_prob(w).sum(-1)

            # The reconstructions are the same
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

            lw = lpx_z + self.beta * (lpz - lqu_x - lqw_x)

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

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod
        if all([s in self.encoders.keys() for s in cond_mod]):
            # For the conditioning modalities we compute all the embeddings
            encoders_outputs = {m: self.encoders[m](inputs.data[m]) for m in cond_mod}

            # Choose one of the conditioning modalities at random to sample the shared information.
            random_mod = np.random.choice(cond_mod)

            # Sample the shared latent code
            mu = encoders_outputs[random_mod].embedding
            log_var = encoders_outputs[random_mod].log_covariance
            sigma = self.log_var_to_std(log_var)

            qz_x = self.post_dist(mu, sigma)
            sample_shape = torch.Size([]) if N == 1 else torch.Size([N])
            z = qz_x.rsample(sample_shape)

            flatten = kwargs.pop("flatten", False)
            if flatten:
                z = z.reshape(-1, self.latent_dim)

            # Modality specific encodings : given by encoders for conditioning modalities
            # Sampling from the priors for the rest of the modalities.

            style_z = dict()

            for m in self.encoders:
                if m not in cond_mod:
                    # Sample from priors parameters.
                    if self.reconstruction_option == "single_prior":
                        mu_m = self.mean_priors[m]
                        logvar_m = self.logvars_priors[m]

                    if self.reconstruction_option == "joint_prior":
                        mu_m = self.mean_priors["shared"][:, self.latent_dim :]
                        logvar_m = self.logvars_priors["shared"][:, self.latent_dim :]

                    mu_m = torch.cat([mu_m] * len(mu), dim=0)
                    logvar_m = torch.cat([logvar_m] * len(mu), dim=0)

                else:
                    # Sample from posteriors parameters
                    mu_m = encoders_outputs[m].style_embedding
                    logvar_m = encoders_outputs[m].style_log_covariance

                sigma_m = self.log_var_to_std(logvar_m)
                style_z[m] = self.post_dist(mu_m, sigma_m).rsample(sample_shape)
                if flatten:
                    style_z[m] = style_z[m].reshape(-1, self.modalities_specific_dim)

            return ModelOutput(z=z, one_latent_space=False, modalities_z=style_z)

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """
        Return the estimated negative log-likelihood summed over the inputs.
        The negative log-likelihood is estimated using importance sampling.

        Args:
            inputs : the data to compute the joint likelihood

        """
        raise (NotImplementedError)

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
