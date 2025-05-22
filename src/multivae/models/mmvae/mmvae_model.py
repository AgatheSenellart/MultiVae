import logging
import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from pythae.models.base.base_utils import ModelOutput
from torch.distributions import Laplace, Normal

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.data.utils import drop_unused_modalities

from ..base import BaseMultiVAE
from .mmvae_config import MMVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MMVAE(BaseMultiVAE):
    """The Variational Mixture-of-Experts Autoencoder model.

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
        """For latent distributions parameters, transform the log covariance to the
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
        """Forward pass of the model. Outputs the loss and metrics."""
        # First compute all the encodings for all modalities

        # drop modalities that are completely unavailable in the batch to avoid Nan in backward
        inputs = drop_unused_modalities(inputs)

        embeddings = {}
        qz_xs = {}
        qz_xs_detach = {}
        reconstructions = {}

        compute_loss = kwargs.pop("compute_loss", True)
        detailed_output = kwargs.pop("detailed_output", False)
        k_iwae = kwargs.pop("K", self.model_config.K)

        for cond_mod in inputs.data:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance

            sigma = self.log_var_to_std(log_var)

            qz_x = self.post_dist(mu, sigma)
            z_x = qz_x.rsample([k_iwae])  # K,n_batch,latent_dim

            # The DREG loss uses detached parameters in the loss computation afterwards.
            qz_x_detach = self.post_dist(mu.detach(), sigma.detach())

            # Then compute all the cross-modal reconstructions
            reconstructions[cond_mod] = {}
            for recon_mod in inputs.data:
                decoder = self.decoders[recon_mod]
                z = z_x.reshape(-1, z_x.shape[-1])  # (K*n_batch, latent_dim)
                recon = decoder(z)["reconstruction"]
                recon = recon.reshape((*z_x.shape[:-1], *recon.shape[1:]))
                reconstructions[cond_mod][recon_mod] = recon

            qz_xs[cond_mod] = qz_x
            embeddings[cond_mod] = z_x
            qz_xs_detach[cond_mod] = qz_x_detach

        # Compute DREG loss
        if compute_loss:
            if self.model_config.loss == "dreg_looser":
                loss_output = self.dreg_looser(
                    qz_xs_detach, embeddings, reconstructions, inputs
                )
            elif self.model_config.loss == "iwae_looser":
                loss_output = self.iwae_looser(
                    qz_xs, embeddings, reconstructions, inputs
                )
            else:
                raise NotImplementedError()

        else:
            loss_output = ModelOutput()

        if detailed_output:
            loss_output["qz_xs"] = qz_xs
            loss_output["qz_xs_detach"] = qz_xs_detach
            loss_output["zss"] = embeddings
            loss_output["recon"] = reconstructions

        return loss_output

    def compute_k_lws(self, qz_xs, embeddings, reconstructions, inputs):
        """Compute likelihood terms for all modalities and for all k.

        returns :
            dict containing the likelihoods terms (not aggregated)
            for all modalities.
        """
        if hasattr(inputs, "masks"):
            # Compute the number of available modalities per sample
            n_mods_sample = torch.sum(
                torch.stack(tuple(inputs.masks.values())).int(), dim=0
            )
        else:
            n_mods_sample = torch.tensor([self.n_modalities])

        lws = {}  # to collect likelihoods

        for mod in embeddings:
            z = embeddings[mod]  # (K, n_batch, latent_dim)
            n_mods_sample = n_mods_sample.to(z.device)
            prior = self.prior_dist(*self.pz_params)

            ### Compute log p(z)
            lpz = prior.log_prob(z).sum(-1)

            ### Compute log q(z|X)

            # Get all the log(q(z|x_i))
            if hasattr(inputs, "masks"):
                # For incomplete data, we only use available modalities
                lqz_x = []
                for m in qz_xs:
                    qz = qz_xs[m].log_prob(z).sum(-1)
                    # Set the probability to 0 for unavailable modalities,
                    # so that they don't weigh in the Mixture-of-Experts
                    qz[torch.stack([inputs.masks[m] == False] * len(z))] = -torch.inf
                    lqz_x.append(qz)

                lqz_x = torch.stack(lqz_x)  # n_modalities,K,nbatch
            else:
                lqz_x = torch.stack(
                    [qz_xs[m].log_prob(z).sum(-1) for m in qz_xs]
                )  # n_modalities,K,nbatch

            # Compute the mixture of expert probability
            lqz_x = torch.logsumexp(lqz_x, dim=0) - torch.log(
                n_mods_sample
            )  # log_mean_exp

            ### Compute log p(X|z)
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
            lw = lpx_z + lpz - lqz_x

            if hasattr(inputs, "masks"):
                # cancel unavailable modalities
                lw *= inputs.masks[mod].float()

            lws[mod] = lw  # K, batch_size

        return lws, n_mods_sample

    def dreg_looser(self, qz_xs, embeddings, reconstructions, inputs):
        """The DreG estimation for IWAE. losses components in lws needs to have been computed on
        **detached** posteriors.

        """
        lws, n_mods_sample = self.compute_k_lws(
            qz_xs, embeddings, reconstructions, inputs
        )

        # Compute all the wk weights for individual likelihoods
        wk = {}
        with torch.no_grad():
            for mod, lw in lws.items():
                wk[mod] = (
                    lw - torch.logsumexp(lw, 0, keepdim=True)
                ).exp()  # K, n_batch

        # Compute the loss
        lws = torch.stack(
            [(lws[mod] * wk[mod]) for mod in embeddings], dim=0
        )  # n_modalities,K, n_batch
        lws = lws.sum(1)  # sum on K

        # The gradient with respect to \phi is multiplied one more time by wk
        # To achieve that, we register a hook on the latent variables z
        for mod in embeddings:
            embeddings[mod].register_hook(
                lambda grad, w=wk[mod]: w.unsqueeze(-1) * grad
            )

        # Take the mean over modalities
        lws = lws.sum(0) / n_mods_sample

        # Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics={})

    def iwae_looser(self, qz_xs, embeddings, reconstructions, inputs):
        """Compute the iwae loss without the DReG estimator for the gradient."""
        lws, n_mods_sample = self.compute_k_lws(
            qz_xs, embeddings, reconstructions, inputs
        )

        # Transform into a tensor
        lws = torch.stack(list(lws.values()), dim=0)  # n_modalities, K, n_batch

        # Take log_mean_exp on K to compute the IWAE estimation
        lws = torch.logsumexp(lws, dim=1) - math.log(
            lws.size(1)
        )  # n_modalities, n_batch

        # Take the mean on modalities outside the log
        lws = lws.sum(0) / n_mods_sample

        # Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics={})

    def iwae(self, qz_xs, embeddings, reconstructions, inputs):
        lws, n_mods_sample = self.compute_k_lws(
            qz_xs, embeddings, reconstructions, inputs
        )

        # Transform into a tensor
        lws = torch.stack(list(lws.values()), dim=0)  # n_modalities, K, n_batch

        # Take log_mean_exp on K to compute the IWAE estimation
        lws = torch.logsumexp(lws, dim=1) - math.log(
            lws.size(1)
        )  # n_modalities, n_batch

        # Take log_mean_exp on the modalities
        lws = torch.logsumexp(lws, dim=0) - n_mods_sample.log()

        # Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics={})

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
            ModelOutput instance with fields 'z' (torch.Tensor (n_data, N, latent_dim)),'one_latent_space' (bool) = True

        """
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

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

    @torch.no_grad()
    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """Estimate the negative joint likelihood.

        Args:
            inputs (MultimodalBaseDataset) : a batch of samples.
            K (int) : the number of importance samples for the estimation. Default to 1000.
            batch_size_K (int) : Default to 100.

        Returns:
            The negative log-likelihood summed over the batch.
        """
        # Check the dataset is not incomplete
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The compute_joint_nll method is not yet implemented for incomplete datasets."
            )

        # Compute all the parameters of the joint posterior q(z|x_1), q(z|x_2), ...
        post_params = []
        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance
            sigma = self.log_var_to_std(log_var)
            post_params.append((mu, sigma))

        # Sample K latents from the joint posterior
        z_joint = self.encode(inputs, N=K).z.permute(1, 0, 2)  # n_data x K x latent_dim
        n_data, _, _ = z_joint.shape

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
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
                prior = self.prior_dist(*self.pz_params)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors ln(q(z|X)) = ln(1/M \sum_m q(z|x_m))
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
        samples and modality rescaling.
        """
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
            lw = -self.iwae(output.qz_xs, output.zss, output.recon, inputs).loss
            lws.append(lw + np.log(n_samples * self.n_modalities))

        ll = torch.logsumexp(torch.stack(lws), dim=0) - np.log(
            nb_computed_samples * self.n_modalities
        )  # n_batch
        return -ll  # we return the negative log liklihood

    def generate_from_prior(self, n_samples, **kwargs):
        sample_shape = [n_samples] if n_samples > 1 else []
        z = self.prior_dist(*self.pz_params).rsample(sample_shape).to(self.device)
        return ModelOutput(z=z.squeeze(), one_latent_space=True)
