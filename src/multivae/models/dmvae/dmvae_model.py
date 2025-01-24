from itertools import combinations
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from numpy.random import choice
from pythae.models.base.base_utils import ModelOutput

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from .dmvae_config import DMVAEConfig


class DMVAE(BaseMultiVAE):
    """
    The DVAE model.

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
        self, model_config: DMVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.modalities_specific_dim = model_config.modalities_specific_dim
        self.beta = model_config.beta
        self.model_name = "MVAE"
        self.set_private_betas(model_config.modality_specific_betas)
    
    def set_private_betas(self, beta_dict):
        if beta_dict is None:
            self.private_betas = {mod : 1.0 for mod in self.encoders}
        else:
            if not self.encoders.keys() == beta_dict.keys():
                raise AttributeError("The modality_specific_betas doesn't have the same "
                                     "keys (modalities) as the provided encoders dict.")
            self.private_betas = beta_dict
    

    def poe(self, mus_list, log_vars_list):
        
        if len(mus_list) == 1:
            return mus_list[0], log_vars_list[0]
        
        mus = mus_list.copy()
        log_vars = log_vars_list.copy()

        # Add the prior to the product of experts
        mus.append(torch.zeros_like(mus[0]))
        log_vars.append(torch.zeros_like(log_vars[0]))

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars])  # Compute the inverse of variances
        lnV = -torch.logsumexp(lnT, dim=0)  # variances of the product of expert
        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        return joint_mu, lnV
    
    def infer_latent_parameters(self, inputs, subset=None):
        # if no subset is provided, use all available modalities
        if subset is None:
            subset = list(inputs.data.keys())
        
        
        # First compute all embeddings 
        private_params = dict()
        shared_params = dict()
        
        for mod in subset:
            output_mod = self.encoders[mod](inputs.data['mod'])
            private_params[mod]=(output_mod.style_embedding, output_mod.style_logcovariance)
            shared_params[mod]=(output_mod.embedding, output_mod.log_covariance)
        
        # Compute the PoE for the shared variable
        list_mu = [shared_params[mod][0] for mod in shared_params]
        list_lvs = []
        
        for mod in subset:
            log_var_mod = shared_params[mod][1]
            if hasattr(inputs, "masks"):
                        log_var_mod[(1 - inputs.masks[mod].int()).bool().flatten()] = (
                            torch.inf
                        )
        
        # For unavailable modalities, set the variance to infinity so that it doesn't count in the PoE
        
        joint_mu, joint_lv=self.poe(list_mu, list_lvs) # N(0,I) prior is added in the function
        
        
        return joint_mu, joint_lv, shared_params, private_params


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
        
        
        joint_mu, joint_lv,shared_params, private_params = self.infer_latent_parameters(inputs)
        
        metrics = dict()
        # Compute the joint elbo
        joint_elbo = self.compute_elbo(joint_mu, joint_lv, private_params,inputs)
        loss = joint_elbo
        metrics['joint']=joint_elbo
        
        # Compute crossmodal elbos
        for m in shared_params:
            mod_elbo = self.compute_elbo(shared_params[m][0], shared_params[m][1], private_params, inputs)
            loss += mod_elbo
            metrics[m] = mod_elbo
        
        return ModelOutput(loss=loss)
    
    
    def kl_divergence(self, mean, log_var, prior_mean, prior_log_var):

        kl = (
            0.5
            * (
                prior_log_var
                - log_var
                - 1
                + torch.exp(log_var - prior_log_var)
                + (mean - prior_mean) ** 2
            )
            / torch.exp(prior_log_var)
        )

        return kl.sum(dim=-1)
    
    def compute_elbo(self,q_mu, q_lv, private_params, inputs):
        
        sigma = torch.exp(0.5*q_lv)
        shared_z = dist.Normal(q_mu, sigma).rsample()
        
        # Compute reconstructions
        recon_loss = 0
        for mod in self.encoders:
            
            # Sample the modality specific 
            mu, sigma = private_params[mod][0], torch.exp(0.5*private_params[mod][1])
            z_mod = dist.Normal(mu, sigma).rsample()
            
            z = torch.cat([
                shared_z, z_mod
            ])
            
            recon_mod = self.decoders[mod](z).reconstruction
            
            recon_loss += self.recon_log_probs(recon_mod, inputs.data['mod']) * self.rescale_factors[mod]
            
        # Compute KL divergence for shared variable
        shared_kl = self.kl_divergence(q_mu, q_lv, torch.zeros_like(q_mu), torch.zeros_like(q_lv))
        
        kl=shared_kl*self.beta
        # Add the modality specific kls
        for mod in self.encoders:
            mu, lv = private_params[mod]
            kl += self.kl_divergence(mu,lv, 
                                     torch.zeros_like(mu), torch.zeros_like(lv))*self.private_betas[mod]
        
        
        neg_elbo = -recon_loss + kl
        
        # TODO: filter unavailable data
        
        return neg_elbo.sum()
        
        
    def encode(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
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

        # Call super to perform some checks and preprocess the cond_mod argument
        # you obtain a list of the modalities' names to condition on
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # Compute the shared latent variable conditioning on input modalities
        sub_mu, sub_logvar,_, private_params = self.compute_mu_log_var_subset(inputs, cond_mod)
        sub_std = torch.exp(0.5 * sub_logvar)
        sample_shape = [N] if N > 1 else []

        return_mean = kwargs.pop("return_mean", False)

        if return_mean:
            z = torch.stack([sub_mu] * N) if N > 1 else sub_mu
        else:
            z = dist.Normal(sub_mu, sub_std).rsample(sample_shape)
        flatten = kwargs.pop("flatten", False)
        if flatten:
            z = z.reshape(-1, self.latent_dim)
        
        modalities_z = dict()
        for mod in cond_mod:
            mod_mu, mod_std = private_params[mod][0], torch.exp(0.5*private_params[mod][1])
            if return_mean:
                mod_z = torch.stack([mod_mu] * N) if N > 1 else mod_mu
            else:
                mod_z = dist.Normal(mod_mu, mod_std).rsample(sample_shape)
            if flatten:
                mod_z = mod_z.reshape(-1, self.latent_dim)
            modalities_z[mod] = mod_z

        return ModelOutput(z=z, one_latent_space=False, modalities_z=modalities_z)

    def compute_joint_nll(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        raise NotImplementedError()
    
        """Computes the joint_negative_nll for a batch of inputs."""

        # iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        n_data = len(inputs.data[list(inputs.data.keys())[0]])
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []

            # Compute the parameters of the joint posterior
            mu, log_var = self.compute_mu_log_var_subset(
                MultimodalBaseDataset(
                    data={k: inputs.data[k][i].unsqueeze(0) for k in inputs.data}
                ),
                list(self.encoders.keys()),
            )

            sigma = torch.exp(0.5 * log_var)
            qz_xy = dist.Normal(mu, sigma)

            # And sample from the posterior
            z_joint = qz_xy.rsample([K]).squeeze()  # shape K x latent_dim

            while start_idx < stop_idx:
                latents = z_joint[start_idx:stop_idx]

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
                qz_xy = dist.Normal(mu.squeeze(), sigma.squeeze())
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
