from itertools import combinations
from multivae.data.datasets.base import MultimodalBaseDataset
from ..base import BaseMultiVAE
from .mvae_config import MVAEConfig
from scipy.special import comb
from numpy.random import choice
import numpy as np
import torch
import torch.distributions as dist
from torch.distributions import kl_divergence

class MVAE(BaseMultiVAE):
    
    def __init__(self, model_config: MVAEConfig, encoders: dict = None, decoders: dict = None):
        super().__init__(model_config, encoders, decoders)
        
        self.k = model_config.k
        self.set_subsets()
        
    
    def set_subsets(self):
        
        self.subsets = []
        for i in range(2,self.n_modalities):
            self.subsets += combinations(list(self.encoders.keys()),r=i)
    
    
    def poe(self, mus_list, log_vars_list):
        
        mus = mus_list.copy()
        log_vars=log_vars_list.copy()
        
        # Add the prior to the product of experts
        mus.append(torch.zeros_like(mus[0]))
        log_vars.append(torch.zeros_like(log_vars[0]))

        # Compute the joint posterior
        lnT = torch.stack([-l for l in log_vars])  # Compute the inverse of variances
        
        lnV = - torch.logsumexp(lnT, dim=0)  # variances of the product of expert
        mus = torch.stack(mus)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        joint_std = torch.exp(0.5 * lnV)
        return joint_mu, joint_std
        
    
    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """The main function of the model that computes the loss and some monitoring metrics.
        One of the advantages of MVAE is that we can train with incomplete data. 

        Args:
            inputs (MultimodalBaseDataset): The data.
            masks (Dict[str,torch.Tensor]): A dictionary containing the information of the missing data.
                For each modality, a boolean tensor indicates which samples are available. (The non 
                available samples are assumed to be replaced with zero values in the multimodal dataset entry.)
                If None is provided, the data is assumed to be complete. 
        """
        
        
        masks = kwargs.pop('masks', None)
        
        # Compute the unimodal elbos
        total_loss = 0
        unimodal_elbos = {}
        mus = {}
        log_vars = {}
        for mod in self.encoders:
            output_mod = self.encoders[mod](inputs.data[mod])
            mu_mod, log_var_mod = output_mod.mu, output_mod.log_var
            sigma_mod = torch.exp(0.5*log_var_mod)
            mus[mod] = mu_mod
            log_vars[mod] = log_var_mod
            
            z_mod = dist.Normal(mu_mod, sigma_mod).rsample()
            
            recon = self.decoders[mod](z_mod).reconstruction
            recon_loss = self.recon_losses[mod](recon, inputs.data[mod])
            kld_mod = self.kl_prior(mu_mod,sigma_mod)
            unimodal_elbos[mod] = recon_loss + kld_mod
            total_loss += recon_loss + kld_mod
            
        # Compute the joint elbo
        joint_mu, joint_log_var = self.poe(zip())
            

    
   
    
    def kl_prior(self,mu, std):
        return kl_divergence(dist.Normal(mu, std), dist.Normal(*self.pz_params)).sum()
    
    
    def infer_latent_from_mod(self, cond_mod, x):
        o = self.vaes[cond_mod].encoder(x)
        mu, log_var = o.embedding, o.log_covariance
        # poe with prior
        mu, std = self.poe([mu],[log_var]) 
        z = dist.Normal(mu, std).rsample()
        return z
              

    def forward(self, x):
        """
            Using encoders and decoders from both distributions, compute all the latent variables,
            reconstructions...
        """

        # Compute the reconstruction terms
        elbo = 0
        
        mus_tilde = []
        lnV_tilde = []
        


        for m, vae in enumerate(self.vaes):
            o = vae.encoder(x[m])
            u_mu, u_log_var = o.embedding, o.log_covariance
            # Save the unimodal embedding
            mus_tilde.append(u_mu)
            lnV_tilde.append(u_log_var)
            
            # Compute the unimodal elbos
            mu, std =  self.poe([u_mu], [u_log_var])
            # print(m, mu, std)
            # mu, std = u_mu, torch.exp(0.5*u_log_var)
            z = dist.Normal(mu, std).rsample()
            recon = vae.decoder(z).reconstruction
            elbo += -1/2*torch.sum((x[m]-recon)**2) * self.lik_scaling[m] - self.kl_prior(mu, std)

        # Add the joint elbo
        joint_mu, joint_std = self.poe(mus_tilde, lnV_tilde)
        z_joint = dist.Normal(joint_mu, joint_std).rsample()

        # Reconstruction term in each modality
        for m, vae in enumerate(self.vaes):
            recon = (vae.decoder(z_joint)['reconstruction'])
            elbo += -1/2*torch.sum((x[m]-recon)**2) * self.lik_scaling[m]
        
        # Joint KL divergence
        elbo -= self.kl_prior(joint_mu, joint_std)
            
        # If using the subsampling paradigm, sample subsets and compute the poe
        if self.subsampling :
            # randomly select k subsets
            subsets = self.subsets[np.random.choice(len(self.subsets), self.k_subsample,replace=False)]
            # print(subsets)

            for s in subsets:
                sub_mus, sub_log_vars = [mus_tilde[i] for i in s], [lnV_tilde[i] for i in s]
                mu, std = self.poe(sub_mus, sub_log_vars)
                sub_z = dist.Normal(mu, std).rsample()
                elbo -= self.kl_prior(mu, std)
                # Reconstruction terms
                for m in s:
                    recon = self.vaes[m].decoder(sub_z).reconstruction
                    elbo += torch.sum(-1 / 2 * (recon - x[m]) ** 2) * self.lik_scaling[m]
                    
            # print('computed subsampled elbos')

        res_dict = dict(
            elbo=elbo,
            z_joint=z_joint,
            joint_mu=joint_mu,
            joint_std=joint_std
        )

        return res_dict
        
        
        