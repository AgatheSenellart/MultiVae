from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from ..base.base_utils import poe, rsample_from_gaussian
from .mvtcae_config import MVTCAEConfig


class MVTCAE(BaseMultiVAE):
    """MVTCAE model.

    Args:
        model_config (MVTCAEConfig): An instance of MVTCAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.


    """

    def __init__(
        self, model_config: MVTCAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.alpha = model_config.alpha
        self.beta = model_config.beta
        self.model_name = "MVTCAE"

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Forward pass of the model that returns the loss."""
        # Compute latents parameters for all subsets
        latents = self._inference(inputs)
        results = {}

        # Sample from the joint posterior
        joint_mu, joint_logvar = latents["joint"][0], latents["joint"][1]
        shared_embeddings = rsample_from_gaussian(joint_mu, joint_logvar)
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

        return ModelOutput(
            loss=total_loss / ndata, loss_sum=total_loss, metrics=results
        )

    def _modality_encode(
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
                output.log_covariance[~inputs.masks[m_key].bool()] = torch.inf
            encoders_outputs[m_key] = output

        return encoders_outputs

    def _inference(self, inputs: MultimodalBaseDataset, **kwargs):
        """This function takes all the modalities contained in inputs
        and compute the product of experts of the modalities encoders.

        Args:
            inputs (MultimodalBaseDataset): The data.

        Returns:
            dict : Contains the modalities' encoders parameters and the poe parameters.
        """
        latents = {}
        enc_mods = self._modality_encode(inputs)
        latents["modalities"] = enc_mods

        device = enc_mods[list(inputs.data.keys())[0]].embedding.device
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

        joint_mu, joint_logvar = poe(mus, logvars)

        latents["joint"] = [joint_mu, joint_logvar]
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

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities
                names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.
            return_mean (bool) : if True, returns the mean of the posterior distribution (instead of a sample).


        Returns:
            ModelOutput instance with fields:
                z (torch.Tensor (n_data, N, latent_dim))
                one_latent_space (bool) = True

        """
        # Call super() function to transform to preprocess cond_mod. You obtain a list of
        # the modalities' names to condition on.
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # Only keep the relevant modalities for prediction
        cond_inputs = MultimodalBaseDataset(
            data={k: inputs.data[k] for k in cond_mod},
        )

        latents_subsets = self._inference(cond_inputs)
        mu, log_var = latents_subsets["joint"]
        flatten = kwargs.pop("flatten", False)
        z = rsample_from_gaussian(
            mu, log_var, N=N, return_mean=return_mean, flatten=flatten
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
        mu, log_var = self._inference(inputs)["joint"]
        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)

        # Sample K latents from the joint posterior
        z_joint = qz_xy.rsample([K]).permute(
            1, 0, 2
        )  # shape :  n_data x K x latent_dim
        n_data, _, _ = z_joint.shape

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            while start_idx < stop_idx:
                latents = z_joint[i][start_idx:stop_idx]

                # Compute p(x_m|z) for z in latents and for each modality m
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

                # Compute posteriors -ln(q(z|X))
                qz_xy = dist.Normal(mu[i], sigma[i])
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch of samples K
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
