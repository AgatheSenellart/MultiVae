import logging
import math
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
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
from .cmvae_config import CMVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

# TODO : code a compute_clusters function / and the post-hoc selection of clusters / adapt for incomplete datasets


class CMVAE(BaseMultiVAE):
    """
    The CMVAE model.

    Args:
        model_config (CMVAEConfig): An instance of MMVAEConfig in which any model's
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
        model_config: CMVAEConfig,
        encoders: dict = None,
        decoders: dict = None,
    ):
        if model_config.modalities_specific_dim is None:
            raise AttributeError(
                "The modalities_specific_dim attribute must"
                " be provided in the model config."
            )

        super().__init__(model_config, encoders, decoders)
        self.model_name = "CMVAE"


        if model_config.prior_and_posterior_dist == "laplace_with_softmax":
            self.latent_dist = Laplace
        elif model_config.prior_and_posterior_dist == "normal":
            self.latent_dist = Normal
        elif model_config.prior_and_posterior_dist == "normal_with_softplus":
            self.latent_dist = Normal
        else:
            raise AttributeError(
                " The posterior_dist parameter must be "
                " either 'laplace_with_softmax','normal' or 'normal_with_softplus'. "
                f" {model_config.prior_and_posterior_dist} was provided."
            )     
        self.multiple_latent_spaces = True
        self.n_clusters = model_config.number_of_clusters
        self.style_dims = {m: self.model_config.modalities_specific_dim for m in self.encoders}
        
        # Set the modality specific priors for private spaces (referred to as r in )
        self.r_mean_priors = torch.nn.ParameterDict()
        self.r_logvars_priors = torch.nn.ParameterDict()

        for mod in list(self.encoders.keys()):
            self.r_mean_priors[mod] = torch.nn.Parameter(
                torch.zeros(1, model_config.modalities_specific_dim),
                requires_grad=False,
            ) # the mean is fixed but the scale can change
            self.r_logvars_priors[mod] = torch.nn.Parameter(
                torch.zeros(1, model_config.modalities_specific_dim),
                requires_grad=model_config.learn_modality_prior,
            )
        
        # Set the regularization prior for the private spaces (referred to as p(w_m)) 
        # in the paper
        
        self.w_mean_prior = torch.nn.Parameter(torch.zeros(1,model_config.modalities_specific_dim), 
                                               requires_grad=False)
        self.w_logvar_prior = torch.nn.Parameter(torch.zeros(1,model_config.modalities_specific_dim), 
                                               requires_grad=False)

        # Initialize the weights for the cluster distribution
        self._pc_params = torch.nn.Parameter(
                torch.zeros(self.n_clusters),
                requires_grad=True,
            )
        
        # Initialize the mean and variances for each cluster in the shared latent spaces
        self.mean_clusters = nn.ParameterList([
            nn.Parameter(((2*torch.rand(1, self.latent_dim))-1), requires_grad=True) for c_k in range(self.n_clusters)
        ])
        # NOTE : the scales are fixed to 1 in the original code !
        self.logvar_clusters = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.latent_dim), False) for c_k in range(self.n_clusters)
        ])

        
        
    @property
    def pc_params(self):
        """

        Returns: Parameters of uniform prior distribution on latent clusters.

        """
        return F.softmax(self._pc_params, dim=-1)


    def log_var_to_std(self, log_var):
        """
        For latent distributions parameters, transform the log covariance to the
        standard deviation of the distribution either applying softmax or softplus.
        This follows the original implementation.
        """

        if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
            return F.softmax(log_var, dim=-1) * log_var.size(-1) + 1e-6
        elif self.model_config.prior_and_posterior_dist == "normal_with_softplus":
            return F.softplus(log_var) + 1e-6
        else:
            return torch.exp(0.5 * log_var)
        
    def compute_posteriors_and_embeddings(self, inputs, detach,**kwargs):

        # Drop unused modalities
        inputs = drop_unused_modalities(inputs)

        # First compute all the encodings for all modalities
        embeddings = {}
        posteriors = {}
        reconstructions = {}

        k_iwae = kwargs.pop("K", self.model_config.K)
        for cond_mod in inputs.data:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance
            mu_style = output.style_embedding
            log_var_style = output.style_log_covariance

            sigma = self.log_var_to_std(log_var)
            sigma_style = self.log_var_to_std(log_var_style)

            # Shared latent variable
            qu_x = self.latent_dist(mu, sigma)
            u_x = qu_x.rsample([k_iwae])

            # Private latent variable
            qw_x = self.latent_dist(mu_style, sigma_style)
            w_x = qw_x.rsample([k_iwae])


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
                        [self.r_mean_priors[recon_mod]] * len(mu), axis=0
                    )
                    sigma_prior_mod = torch.cat(
                        [self.log_var_to_std(self.r_logvars_priors[recon_mod])] * len(mu),
                        axis=0,
                    )

                    w = self.latent_dist(
                        mu_prior_mod,
                        sigma_prior_mod,
                    ).rsample([k_iwae]) # K, n_batch, modality_specific_sim
                    
                    z_x = torch.cat([u_x, w], dim=-1)
                    
                # Decode
                z = z_x.reshape(-1, z_x.shape[-1])
                recon = self.decoders[recon_mod](z)["reconstruction"]
                recon = recon.reshape((*z_x.shape[:-1], *recon.shape[1:]))

                reconstructions[cond_mod][recon_mod] = recon
                
            # The DREG loss uses detached posteriors in the loss computation afterwards.
            if detach:
                qu_x = self.latent_dist(mu.clone().detach(), sigma.clone().detach())
                qw_x = self.latent_dist(
                mu_style.clone().detach(), sigma_style.clone().detach()
            )
            
            posteriors[cond_mod] = {'u':qu_x,'w':qw_x}
            embeddings[cond_mod] = {'u':u_x, 'w':w_x}

        return posteriors, embeddings, reconstructions


    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        
        if self.model_config.loss == "dreg_looser":
            posteriors, embeddings, reconstructions = self.compute_posteriors_and_embeddings(inputs,detach=True,**kwargs)
            # For the DreG estimation, we compute the individual likelihoods with detached posteriors.
            lws, embeddings, n_mods_sample = self.compute_k_lws(posteriors,embeddings,reconstructions, inputs)
            return self.dreg_looser(lws,embeddings, n_mods_sample)
            
        if self.model_config.loss == "iwae_looser":
            posteriors, embeddings, reconstructions = self.compute_posteriors_and_embeddings(inputs,detach=False,**kwargs)
            lws,_,n_mods_sample= self.compute_k_lws(posteriors,embeddings, reconstructions, inputs)
        
            return self.iwae_looser(lws, n_mods_sample)
            
        raise NotImplementedError()
        



    def compute_k_lws(self, posteriors, embeddings, reconstructions, inputs):
        """
        
        Compute all losses components without any aggregation on K nor batch
        
        Returns:

            lws (dict)
            embeddings (dict)
            n_mod_samples
        
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
            
            ### Compute log p(w_m) / regularizing prior for the private spaces
            mu = self.w_mean_prior
            sigma = self.log_var_to_std(self.w_logvar_prior)
            lpw = self.latent_dist(mu, sigma).log_prob(embeddings[mod]["w"]).sum(-1)
            
            ### Compute log q(w_m | x_m)
            lqw_x = posteriors[mod]['w'].log_prob(embeddings[mod]["w"]).sum(-1)
            
            ### Compute log q_{\phi_z}(z| X )
            u = embeddings[mod]["u"] # shared latent variable
            if hasattr(inputs, "masks"):
                lqu_x = []
                for m in posteriors:
                    lqu = posteriors[m]['u'].log_prob(u).sum(-1)
                    lqu[torch.stack([inputs.masks[m] == False] * len(u))] = -torch.inf
                    lqu_x.append(lqu)
                lqu_x = torch.stack(lqu_x)
            else:
                lqu_x = torch.stack(
                        [posteriors[m]['u'].log_prob(u).sum(-1) for m in posteriors]
                    )  # n_modalities,K,nbatch
            lqu_x = torch.logsumexp(lqu_x, dim=0) - torch.log(
                n_mods_sample
            ).to(lqu_x.device)  # log_mean_exp
            
            ### Compute log p_{\pi}(c) for all clusters
            
            lpc = torch.log(self.pc_params) # n_clusters
            
            ### Compute log p(z|c) for all clusters
            
            lpzc = []
            for i in range(self.n_clusters):
                mu_cluster = self.mean_clusters[i]
                sigma_cluster = self.log_var_to_std(self.logvar_clusters[i])
                lpzc.append(self.latent_dist(mu_cluster, sigma_cluster).log_prob(u))
            lpzc = torch.stack(lpzc, dim=0) # n_clusters, K, batch_size, latent_dim
            lpzc = lpzc.sum(-1) # n_clusters, K, batch_size
            
            ### Compute q (c | z) for all clusters
            qzc = torch.softmax(lpc.view(self.n_clusters, 1, 1)+lpzc, dim=0) + 1e-20 # shape n_clusters, K, batch_size


            ### Compute \sum_m log p(x_m|z,w_m)
            lpx_z = 0
            for recon_mod in reconstructions[mod]:
                x_recon = reconstructions[mod][recon_mod]
                k_iwae, n_batch = x_recon.shape[0], x_recon.shape[1]
                lpx_z_mod = (
                    self.recon_log_probs[recon_mod](x_recon, inputs.data[recon_mod])
                    .view(k_iwae, n_batch, -1)
                    .mul(self.rescale_factors[recon_mod])
                    .sum(-1)
                )

                if hasattr(inputs, "masks"):
                    # don't reconstruct unavailable modalities
                    lpx_z_mod *= inputs.masks[recon_mod].float()

                lpx_z += lpx_z_mod

            ### Compute the explicit expectation on q(c|z, X)
            lw = 0
            for c,q_c in enumerate(qzc):
                lw_c = lpx_z + self.model_config.beta * (lpc[c] + lpzc[c] + lpw - lqu_x - lqw_x - q_c.log())
                lw += q_c * lw_c
            assert lw.shape[0] ==(k_iwae)
            # lw.shape : (K, n_batch)
            

            if hasattr(inputs, "masks"):
                # cancel unavailable modalities
                lw *= inputs.masks[mod].float()

            lws[mod] = lw
        

        return lws, embeddings,n_mods_sample

    def iwae_looser(self, lws,n_mods_sample):
        """
        The IWAE loss with the sum outside of the log for increased stability.
        (following Shi et al 2019)

        """
        
        lws = torch.stack(list(lws.values()), dim=0) # n_modalities, K, n_batch

        # Take log_mean_exp on K
        lws = torch.logsumexp(lws, dim=1) - math.log(lws.size(1)) # n_modalities, n_batch
        
        # Take the mean on modalities
        lws = lws.sum(0) /n_mods_sample # n_batch
        
        # Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics=dict())
    
    def dreg_looser(self,lws, embeddings, n_mods_sample):
        
        """The DreG estimation for IWAE. losses components in lws needs to have been computed on
        **detached** posteriors.
         
        """
        wk = {}
        with torch.no_grad():
            for mod, lw in lws.items():
                wk[mod] = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp() # K, n_batch
            # wk is a constant that will not require grad
            
        # Compute the loss
        lws = torch.stack([(lws[mod] * wk[mod]) for mod in embeddings],dim=0) # n_modalities,K, n_batch
        lws = lws.sum(1) # sum on K
        
        # The gradient with respect to \phi is multiplied one more time by wk
        # To achieve that, we register a hook on the latent variables u and w
        for mod in embeddings:
            embeddings[mod]["w"].register_hook(lambda grad,w=wk[mod] : w.unsqueeze(-1)*grad)
            embeddings[mod]["u"].register_hook(lambda grad,w=wk[mod] : w.unsqueeze(-1)*grad)
        
        # Average over modalities
        lws = lws.sum(0)/n_mods_sample # n_batch
        
                
        # Return the sum over the batch
        return ModelOutput(loss=-lws.sum(), loss_sum=-lws.sum(), metrics=dict())
        
        

    def encode(
        self,
        inputs: MultimodalBaseDataset,
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
                one_latent_space (bool) = False
                modalities_z (Dict[str,torch.Tensor (n_data, N, latent_dim) ])



        """

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
            
            # Adapt shape in the case of one sample for uniformity
            if len(mu.shape)==1:
                mu = mu.unsqueeze(0)
                sigma = sigma.unsqueeze(0)

            qz_x = self.latent_dist(mu, sigma)
            sample_shape = torch.Size([]) if N == 1 else torch.Size([N])
            z = qz_x.rsample(sample_shape)

            flatten = kwargs.pop("flatten", False)
            
            
            if flatten:
                z = z.reshape(-1, self.latent_dim)

            # Modality specific encodings : given by encoders for conditioning modalities
            # Sampling from the priors for the rest of the modalities.

            style_z = {}

            for m in self.encoders:
                if m not in cond_mod:
                    # Sample from priors parameters.
                    if self.model_config.reconstruction_option == "single_prior":
                        mu_m = self.r_mean_priors[m]
                        logvar_m = self.r_logvars_priors[m]

                    if self.model_config.reconstruction_option == "joint_prior":
                        mu_m = self.w_mean_prior
                        logvar_m = self.w_logvar_prior

                    mu_m = torch.cat([mu_m] * len(mu), dim=0)
                    logvar_m = torch.cat([logvar_m] * len(mu), dim=0)

                else:
                    # Sample from posteriors parameters
                    mu_m = encoders_outputs[m].style_embedding
                    logvar_m = encoders_outputs[m].style_log_covariance
                    
                if len(mu_m.shape)==1: # eventually adapt the shape when there is one sample for uniformity
                    mu_m = mu_m.unsqueeze(0)
                    logvar_m = logvar_m.unsqueeze(0)

                sigma_m = self.log_var_to_std(logvar_m)
                style_z[m] = self.latent_dist(mu_m, sigma_m).rsample(sample_shape)
                if flatten:
                    style_z[m] = style_z[m].reshape(-1, self.model_config.modalities_specific_dim)

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


    def generate_from_prior(self, n_samples, **kwargs):
        
        # generate the clusters assignements
        
        clusters = dist.Categorical(logits = self._pc_params).sample([n_samples]) # n_samples, n_clusters
        
        # get means for each clusters
        means = torch.cat([self.mean_clusters[c] for c in clusters], dim=0)
        lvs = torch.cat([self.logvar_clusters[c] for c in clusters], dim=0) # n_samples, latent_dims
        
        # sample shared latent variable
        z_shared = self.latent_dist(means, self.log_var_to_std(lvs)).sample() #n_samples,latent_dim
        
        # generate private parameters
        style_z = {}
        for m in self.encoders:
            if self.model_config.reconstruction_option == "single_prior":
                mu_m = self.r_mean_priors[m]
                logvar_m = self.r_logvars_priors[m]

            elif self.model_config.reconstruction_option == "joint_prior":
                mu_m = self.w_mean_prior
                logvar_m = self.w_logvar_prior
            
            else :
                raise NotImplementedError()

            mu_m = torch.cat([mu_m] * n_samples, dim=0)
            logvar_m = torch.cat([logvar_m] * n_samples, dim=0)
            style_z[m] = self.latent_dist(mu_m, self.log_var_to_std(logvar_m)).sample()
        
        return ModelOutput(z=z_shared, one_latent_space = False,modalities_z=style_z)
    
    def predict_clusters(self, 
                         inputs: MultimodalBaseDataset, **kwargs):
        
        """ Returns the clusters for all samples in inputs"""
        
        modalities_cluster_assign = []
        # First we compute the cluster assignements according to each modality individually
        for mod in inputs.data:
            # Compute shared embeddings
            output_encoder = self.encoders[mod](inputs.data[mod])
            mu = output_encoder.embedding
            sigma = self.log_var_to_std(output_encoder.log_covariance)
            
            z = self.latent_dist(mu, sigma).sample()
            
            # Compute p(c|z) \propto p(z|c)p(c)
            lpc = torch.log(self.pc_params + 1e-20) # n_clusters
            lpz_c = [
                self.latent_dist(self.mean_clusters[i], 
                                self.log_var_to_std(self.logvar_clusters[i])).log_prob(z).sum(-1)
                
                for i in range(self.n_clusters)
            ]
            lpz_c = torch.stack(lpz_c, dim=0) # n_clusters, batch_size
            pc_z = torch.softmax(lpc.view(-1,1)+lpz_c, dim=0)#n_clusters, batch_size
            
            cluster_assign = torch.argmax(pc_z,dim=0) #batch_size
            modalities_cluster_assign.append(cluster_assign)

        # Take a majority vote among modalities
        modalities_cluster_assign = torch.stack(modalities_cluster_assign, dim=-1)#batch_size, n_modalities
        vote_cluster = torch.mode(modalities_cluster_assign, dim=-1)[0] #batch_size
        
        return ModelOutput(clusters = vote_cluster)
        
        


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
