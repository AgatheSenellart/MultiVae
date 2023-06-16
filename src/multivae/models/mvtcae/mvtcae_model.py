from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from .mvtcae_config import MVTCAEConfig


class MVTCAE(BaseMultiVAE):

    """

    Implementation for 'Multi-View Representation Learning via Total Correlation Objective'.
    Hwang et al, 2021.

    This code is heavily based on the official implementation that can be found here :
    https://github.com/gr8joo/MVTCAE.


    """

    def __init__(
        self, model_config: MVTCAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.alpha = model_config.alpha
        self.beta = model_config.beta
        self.model_name = "MVTCAE"

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        z = dist.Normal(mu, std).rsample()
        return z

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        # Compute latents parameters for all subsets
        latents = self.inference(inputs)
        results = dict()

        # Sample from the joint posterior
        joint_mu, joint_logvar = latents["joint"][0], latents["joint"][1]
        shared_embeddings = self.reparameterize(joint_mu, joint_logvar)
        ndata = len(shared_embeddings)
        joint_kld = -0.5 * torch.sum(
            1 - joint_logvar.exp() - joint_mu.pow(2) + joint_logvar
        )
        assert not torch.isnan(joint_kld)
        results["joint_divergence"] = joint_kld

        # Compute the reconstruction losses for each modality
        loss_rec = 0
        for m_key in self.encoders.keys():
            # reconstruct this modality from the shared embeddings representation
            recon = self.decoders[m_key](shared_embeddings).reconstruction
            m_rec = (
                -self.recon_log_probs[m_key](recon, inputs.data[m_key])
                * self.rescale_factors[m_key]
            )
            m_rec = m_rec.reshape(m_rec.size(0), -1).sum(-1)

            # Keep only the available samples
            if hasattr(inputs, "masks"):
                m_rec = inputs.masks[m_key].float() * m_rec

            results[m_key] = m_rec.sum()
            loss_rec += m_rec.sum()
        assert not torch.isnan(loss_rec)

        latent_modalities = latents["modalities"]
        kld_losses = 0.0
        for m_key in latent_modalities.keys():
            o = latent_modalities[m_key]
            mu, logvar = o.embedding, o.log_covariance
            results["kld_" + m_key] = -0.5 * (
                1
                - joint_logvar.exp() / logvar.exp()
                - (joint_mu - mu).pow(2) / logvar.exp()
                + joint_logvar
                - logvar
            ).reshape(mu.size(0), -1).sum(-1)

            # Keep only the available samples
            if hasattr(inputs, "masks"):
                results["kld_" + m_key][(1 - inputs.masks[m_key].int()).bool()] = 0

            results["kld_" + m_key] = results["kld_" + m_key].sum()

            kld_losses += results["kld_" + m_key].sum()
        assert not torch.isnan(kld_losses)

        rec_weight = (self.n_modalities - self.alpha) / self.n_modalities
        cvib_weight = self.alpha / self.n_modalities  # 1/6
        vib_weight = 1 - self.alpha  # 0.1

        kld_weighted = cvib_weight * kld_losses + vib_weight * joint_kld
        total_loss = rec_weight * loss_rec + self.beta * kld_weighted

        return ModelOutput(loss=total_loss / ndata, metrics=results)

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
        for m, m_key in enumerate(inputs.data.keys()):
            input_modality = inputs.data[m_key]
            output = self.encoders[m_key](input_modality)
            # For unavailable samples, set the log-variance to infty so that they don't contribute to the
            # product of experts
            if hasattr(inputs, "masks"):
                output.log_covariance[
                    (1 - inputs.masks[m_key].int()).bool()
                ] = torch.inf
            encoders_outputs[m_key] = output

        return encoders_outputs

    def poe(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1.0 / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar

    def ivw_fusion(self, mus: torch.Tensor, logvars: torch.Tensor, weights=None):
        mu_poe, logvar_poe = self.poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def _filter_inputs_with_masks(
        self, inputs: IncompleteDataset, subset: Union[list, tuple]
    ):
        """
        Returns a filtered dataset containing only the samples that are available
        in all the modalities contained in subset.
        The dataset that is returned only contains the modalities in subset.
        """

        filter = torch.tensor(
            True,
        ).to(inputs.masks[subset[0]].device)
        for mod in subset:
            filter = torch.logical_and(filter, inputs.masks[mod])

        filtered_inputs = MultimodalBaseDataset(
            data={k: inputs.data[k][filter] for k in subset},
        )
        return filtered_inputs, filter

    def inference(self, inputs: MultimodalBaseDataset, **kwargs):
        """
        This function takes all the modalities contained in inputs
        and compute the product of experts of the modalities encoders.

        Args:
            inputs (MultimodalBaseDataset): The data.

        Returns:
            dict : Contains the modalities' encoders parameters and the poe parameters.
        """

        latents = dict()
        enc_mods = self.modality_encode(inputs)
        latents["modalities"] = enc_mods

        device = inputs.data[list(inputs.data.keys())[0]].device
        mus = torch.Tensor().to(device)
        logvars = torch.Tensor().to(device)
        mods = list(inputs.data.keys())

        for m, mod in enumerate(mods):
            mus_mod = enc_mods[mod].embedding
            log_vars_mod = enc_mods[mod].log_covariance

            mus = torch.cat((mus, mus_mod.unsqueeze(0)), dim=0)

            logvars = torch.cat((logvars, log_vars_mod.unsqueeze(0)), dim=0)

        # Case with only one sample : adapt the shape
        if len(mus.shape) == 2:
            mus = mus.unsqueeze(1)
            logvars = logvars.unsqueeze(1)

        joint_mu, joint_logvar = self.ivw_fusion(mus, logvars)

        latents["joint"] = [joint_mu, joint_logvar]
        return latents

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ) -> ModelOutput:
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # Only keep the relevant modalities for prediction
        cond_inputs = MultimodalBaseDataset(
            data={k: inputs.data[k] for k in cond_mod},
        )

        latents_subsets = self.inference(cond_inputs)
        mu, log_var = latents_subsets["joint"]
        sample_shape = [N] if N > 1 else []

        return_mean = kwargs.pop("return_mean", False)
        if return_mean:
            z = torch.stack([mu] * N) if N > 1 else mu
        else:
            z = dist.Normal(mu, torch.exp(0.5 * log_var)).rsample(sample_shape)
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
        self.eval()

        # Compute the parameters of the joint posterior
        mu, log_var = self.inference(inputs)["joint"]

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)
        # And sample from the posterior
        z_joint = qz_xy.rsample([K])  # shape K x n_data x latent_dim
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
