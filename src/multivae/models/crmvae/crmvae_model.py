from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from ..base.base_utils import kl_divergence
from .crmvae_config import CRMVAEConfig


class CRMVAE(BaseMultiVAE):
    """

    Main class for the CRMVAE model.


    """

    def __init__(
        self, model_config: CRMVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.model_name = "CRMVAE"

    def _logits_to_scale(self, logits):
        return torch.exp(0.5*logits), logits

    def _rsample(self,latent_params: ModelOutput, size=None):
        mean = latent_params.embedding
        scale, log_var = self._logits_to_scale(latent_params.log_covariance)
        if size is None:
            return mean, log_var, dist.Normal(mean, scale).rsample()
        return mean, log_var, dist.Normal(mean, scale).rsample(size)


    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:

        # Compute latents parameters for q(z|x_i) and q(z|X)
        latents = self._infer_all_latent_parameters(inputs)
        results = {}
        z_samples = {} # for reconstruction
        
        # Sample from q(z|X)
        joint_mu, joint_logvar, shared_embeddings = self._rsample(latents['joint'])
        z_samples['joint'] = shared_embeddings

        # Compute the KL(q(z|X)|p(z)). p(z) is a normal distribution N(0,I)
        joint_kld = kl_divergence(joint_mu, joint_logvar,torch.zeros_like(joint_mu), torch.zeros_like(joint_logvar))
        results["joint_divergence"]  = joint_kld.mean()
        divergence = joint_kld

        # Sample from each unimodal posterior q(z|x_i) and compute KL(q(z|X)|q(z|x_i))
        for m, latent_params in latents["modalities_no_mask"].items():
            mu, log_var, embeddings = self._rsample(latent_params)
            z_samples[m]=embeddings

            # Compute KL(q(z|X) | q(z|x_i))
            results[f'kl_{m}'] = kl_divergence(joint_mu, joint_logvar, mu, log_var)

            # Remove unavailable samples
            if hasattr(inputs, "masks"):
                results[f'kl_{m}'] = results[f'kl_{m}'] * inputs.masks[m].float()

            divergence += results[f'kl_{m}']
            results[f'kl_{m}'] = results[f'kl_{m}'].mean()

        # Compute E_{q(z|X)} log p(x_i|z) + E_{q(z|x_i)}(log p(x_i|z))
        loss_rec = 0
        for gen_mod, decoder in self.decoders.items():

            for m in ['joint', gen_mod]:
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
                if hasattr(inputs, "masks") :
                    m_rec = inputs.masks[gen_mod].float() * m_rec

                loss_rec += m_rec
                #Save metric for monitoring
                results[f'recon_{gen_mod}_from_{m}'] = m_rec.mean()

        # Average accross the posteriors : TODO (average over available posteriors only ?)
        loss_rec = loss_rec /(2*(self.n_modalities + 1))
        divergence = divergence / (self.n_modalities + 1)

        total_loss = loss_rec + self.model_config.beta * divergence
        

        return ModelOutput(
            loss=total_loss.mean(), loss_sum=total_loss.sum(), metrics=results
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
            encoders_outputs[m]= output
            masked_outputs[m] = ModelOutput(embedding = output.embedding, log_covariance = output.log_covariance.clone())
            if hasattr(inputs, "masks"):
                masked_outputs[m].log_covariance[(1 - inputs.masks[m].int()).bool()] = (
                    torch.inf
                )
        return encoders_outputs, masked_outputs

    def _poe(self, mu, logvar, eps=1e-8):
        """Compute the Product of experts of the gaussian distributions."""
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1.0 / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


    def _infer_all_latent_parameters(self, inputs: MultimodalBaseDataset, **kwargs):
        """
        This function takes all the modalities contained in inputs
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
        joint_mu, joint_logvar = self._poe(mus, logvars)

        latents["joint"] = ModelOutput(embedding = joint_mu, log_covariance=joint_logvar)
        return latents

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ) -> ModelOutput:
        """
        Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities
                names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.

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
        # Compute the product of experts of the modalities in cond_mod
        latents_subsets = self._infer_all_latent_parameters(cond_inputs)
        sample_shape = [N] if N > 1 else []

        mu, log_var, z = self._rsample(latents_subsets['joint'], size=sample_shape)

        return_mean = kwargs.pop("return_mean", False)
        if return_mean:
            z = torch.stack([mu] * N) if N > 1 else mu
        
        flatten = kwargs.pop("flatten", False)
        if flatten:
            z = z.reshape(-1, self.latent_dim)

        return ModelOutput(z=z, one_latent_space=True)

   