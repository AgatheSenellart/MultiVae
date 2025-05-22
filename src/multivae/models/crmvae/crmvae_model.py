from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from ..base.base_utils import kl_divergence, poe, rsample_from_gaussian
from .crmvae_config import CRMVAEConfig


class CRMVAE(BaseMultiVAE):
    """Main class for the CRMVAE model proposed in https://openreview.net/forum?id=Rn8u4MYgeNJ.

    Args:
        model_config (CRMVAEConfig): An instance of CRMVAEConfig containing
            all the parameters for the model.
        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.
        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.
    """

    def __init__(
        self, model_config: CRMVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.model_name = "CRMVAE"

    def _rsample(
        self, latent_params: ModelOutput, N=1, return_mean=False, flatten=False
    ):
        mean = latent_params.embedding
        log_var = latent_params.log_covariance
        z = rsample_from_gaussian(mean, log_var, N, return_mean, flatten)
        return mean, log_var, z

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Forward pass of the model. Returns the loss and additional metrics
        in a ModelOutput Instance.
        """
        # Compute latents parameters for q(z|x_i) and q(z|X)
        latents = self._infer_all_latent_parameters(inputs)
        results = {}
        z_samples = {}  # for reconstruction

        # Sample from q(z|X)
        joint_mu, joint_logvar, shared_embeddings = self._rsample(latents["joint"])
        z_samples["joint"] = shared_embeddings

        # Compute the KL(q(z|X)|p(z)). p(z) is a normal distribution N(0,I)
        joint_kld = kl_divergence(
            joint_mu,
            joint_logvar,
            torch.zeros_like(joint_mu),
            torch.zeros_like(joint_logvar),
        )
        results["joint_divergence"] = joint_kld.mean()
        divergence = joint_kld

        # Sample from each unimodal posterior q(z|x_i) and compute KL(q(z|X)|q(z|x_i))
        for m, latent_params in latents["modalities_no_mask"].items():
            mu, log_var, embeddings = self._rsample(latent_params)
            z_samples[m] = embeddings

            # Compute KL(q(z|X) | q(z|x_i))
            results[f"kl_{m}"] = kl_divergence(joint_mu, joint_logvar, mu, log_var)

            # Remove unavailable samples
            if hasattr(inputs, "masks"):
                results[f"kl_{m}"] = results[f"kl_{m}"] * inputs.masks[m].float()

            divergence += results[f"kl_{m}"]
            results[f"kl_{m}"] = results[f"kl_{m}"].mean()

        # Compute E_{q(z|X)} log p(x_i|z) + E_{q(z|x_i)}(log p(x_i|z))
        loss_rec = 0
        for gen_mod, decoder in self.decoders.items():
            for m in ["joint", gen_mod]:
                # for m in ['joint']:
                z = z_samples[m]

                recon = decoder(z).reconstruction
                # apply rescaling factors is any are available
                m_rec = (
                    -self.recon_log_probs[gen_mod](recon, inputs.data[gen_mod])
                    * self.rescale_factors[gen_mod]
                )
                m_rec = m_rec.reshape(m_rec.size(0), -1).sum(-1)

                # Cancel out unavailable samples in the reconstruction
                if hasattr(inputs, "masks"):
                    m_rec = inputs.masks[gen_mod].float() * m_rec

                loss_rec += m_rec
                # Save metric for monitoring
                results[f"recon_{gen_mod}_from_{m}"] = m_rec.mean()

        # Average accross the posteriors : TODO (average over available posteriors only ?)
        loss_rec = loss_rec / (2 * (self.n_modalities + 1))
        divergence = divergence / (self.n_modalities + 1)

        total_loss = loss_rec + self.model_config.beta * divergence

        return ModelOutput(
            loss=total_loss.sum(), loss_sum=total_loss.sum(), metrics=results
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
        encoders_outputs = {}
        masked_outputs = {}
        for m, data_m in inputs.data.items():
            output = self.encoders[m](data_m)
            # For unavailable samples, set the log-variance to infty so that they don't contribute to the
            # product of experts
            encoders_outputs[m] = output
            masked_outputs[m] = ModelOutput(
                embedding=output.embedding, log_covariance=output.log_covariance.clone()
            )
            if hasattr(inputs, "masks"):
                masked_outputs[m].log_covariance[
                    (1 - inputs.masks[m].int()).bool()
                ] = torch.inf
        return encoders_outputs, masked_outputs

    def _infer_all_latent_parameters(self, inputs: MultimodalBaseDataset, **kwargs):
        """This function takes all the modalities contained in inputs
        and compute the product of experts of the modalities encoders.

        Args:
            inputs (MultimodalBaseDataset): The data.

        Returns:
            dict : Contains the modalities' encoders parameters and the poe parameters.
        """
        latents = {}
        enc_mods, masked_mods = self._modality_encode(inputs)
        latents["modalities_no_mask"] = enc_mods
        latents["modalities_masked"] = masked_mods

        device = enc_mods[list(inputs.data.keys())[0]].embedding.device
        mus = torch.Tensor().to(device)
        logvars = torch.Tensor().to(device)

        #  Concatenate all the masked params for computing the PoE
        for masked_params in masked_mods.values():
            mus_mod = masked_params.embedding
            log_vars_mod = masked_params.log_covariance

            mus = torch.cat((mus, mus_mod.unsqueeze(0)), dim=0)
            logvars = torch.cat((logvars, log_vars_mod.unsqueeze(0)), dim=0)

        # Case with only one sample : adapt the shape
        if len(mus.shape) == 2:
            mus = mus.unsqueeze(1)
            logvars = logvars.unsqueeze(1)

        # Compute the PoE and save the result
        joint_mu, joint_logvar = poe(mus, logvars)

        latents["joint"] = ModelOutput(embedding=joint_mu, log_covariance=joint_logvar)
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
        flatten = kwargs.pop("flatten", False)

        # Only keep the relevant modalities for prediction
        cond_inputs = MultimodalBaseDataset(
            data={k: inputs.data[k] for k in cond_mod},
        )
        # Compute the product of experts of the modalities in cond_mod
        latents_subsets = self._infer_all_latent_parameters(cond_inputs)

        _, _, z = self._rsample(
            latents_subsets["joint"], N=N, return_mean=return_mean, flatten=flatten
        )
        return ModelOutput(z=z, one_latent_space=True)

    @torch.no_grad()
    def compute_joint_nll(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ) -> torch.Tensor:
        """Estimate the negative joint likelihood.

        Args:
            inputs (MultimodalBaseDataset) : a batch of samples.
            K (int) : the number of importance samples for the estimation. Default to 1000.
            batch_size_K (int) : Default to 100.

        Returns:
            The negative log-likelihood summed over the batch.
        """
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The compute_joint_nll method is not yet implemented for incomplete datasets."
            )
        # Compute the parameters of the joint posterior
        joint_params = self._infer_all_latent_parameters(inputs)["joint"]

        # Sample K latents from the joint posterior
        mu, logvar, z_joint = self._rsample(joint_params, N=K)
        z_joint = z_joint.permute(1, 0, 2)  # shape :  n_data x K x latent_dim
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
                qz_xy = dist.Normal(mu[i], torch.exp(logvar[i] * 0.5))
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch of samples K
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
