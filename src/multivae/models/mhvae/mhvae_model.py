from typing import Union
from itertools import combinations
import logging

import torch
import torch.distributions as dist

from multivae.data.datasets.base import MultimodalBaseDataset
from multivae.models.base import BaseMultiVAE
from multivae.models.base.base_model import ModelOutput
from multivae.models.nn.base_architectures import BaseEncoder
from .mhvae_config import MHVAEConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MHVAE(BaseMultiVAE):
    """
    
    Multimodal Hierarchical Variational Autoencoder from 
    'Unified Brain MR-Ultrasound Synthesis using Multi-Modal Hierarchical Representations' 
    (Dorent et al, 2O23) (https://arxiv.org/abs/2309.08747)

    The MHVAE is a hierarchical VAE that can handle multiple modalities.
    The latent variable is partitioned into disjoint groups :math:`z = {z_1, z_2, ..., z_L}`
    where L is the number of levels.
    
    The prior on the latent variables is defined as :math:`p_{\theta}(z) = p_{\theta_L}(z_L)\prod_l p_{\theta_l}(z_l|z>l)`
    where :math:`z>l` denotes the latent variables at levels higher than l.
    
    The posterior is defined as :math:`q_{\phi}(z|x) = \prod_l q_{\phi_l}(z_l|x,z>l)`
    that approximates the intractable true posterior :math:`p_{\theta}(z|x)`.
    
    At each level l, the posterior :math: `q_{\phi_l}(z_l|x,z>l)` is approximated by a Product-of-Experts.
    At the deepest level, :math:`q_{\phi_L}(z_L|x) = p_{\theta_L}(z_L) \prod_i q_{\phi_L^{i}}(z_L|x_i)`    
    At following levels, :math:`q_{\phi_l}(z_l|x,z>l) = p_{\theta_l}(z_l|z>l) \prod_i q_{\phi_l^{i}}(z_l|x_i,z>l)` 
    
    Some weights are shared between the different posteriors and priors distribution. 
    To allow flexibility while remaining close to the original implementation, we describe customizable 
    blocks in diagram below. (adaptated from the diagram in the original paper)
    
    .. image:: ./mhvae_architectures.png
    """

    def __init__(self,
                 model_config:MHVAEConfig, 
                 encoders:dict, 
                 decoders:dict,
                 bottom_up_blocks:dict,
                 top_down_blocks:list,
                 posterior_blocks:Union[list,dict],
                 prior_blocks: list
                 ):
        
        """MHVAE model.
        
        Parameters :
        
            model_config (MHVAEConfig) : the model configuration.
            encoders (Dict[str,BaseEncoder]) : contains the first layer encoder per modality.
            decoders (Dict[str, BaseDecoder]) : contains the last layer decoder per modality.
            bottom_up_blocks (Dict[str, list]) : For each modality, contains the (n_latent-1) bottom-up layers. 
                Each layer must be an instance of nn.Module. The last layer must be an instance 
                of BaseEncoder and must return the mean and log_covariance for the deepest latent variable.
            top_down_blocks (List[nn.Module]): contains the (n_latent-1) top-down layers. 
                Each layer must be an instance of nn.Module. 
            posterior_blocks (List or Dict): contains the (n_latent - 1) posterior layers for each modality. 
                Each layer must be an instance of BaseEncoder. The input dimension of each posterior
                block must match 2 * the output dimension of the corresponding top_down_blocks. Provide a list
                if the weights are shared between modalities, and a dictionary if they are not.
            prior_blocks (List): contains the (n_latent - 1) prior layers. 
                Each layer must be an instance of BaseEncoder. The input dimension of each prior
                block must match the output dimension of the corresponding top_down_blocks. 
            
        """
        
        
        # Super method sets up the base fields as well as encoders / decoders
        
        super().__init__(model_config,encoders,decoders)
        self.n_latent = model_config.n_latent
        self.beta = model_config.beta
        self.model_name = 'MHVAE'

        self.sanity_check_bottom_up(encoders, bottom_up_blocks)
        self.set_bottom_up_blocks(bottom_up_blocks)
        
        self.sanity_check_top_down_blocks(top_down_blocks)
        self.set_top_down_blocks(top_down_blocks)
        
        self.sanity_check_prior_blocks(prior_blocks)
        self.prior_blocks = torch.nn.ModuleList(prior_blocks)
        
        self.check_and_set_posterior_blocks(posterior_blocks)
        
        self.model_config.custom_architectures.extend([
            'bottom_up_blocks',
            'top_down_blocks',
            'prior_blocks',
            'posterior_blocks'
        ])
    
    
    def poe(self, mus_list, log_vars_list):
        """Compute the product of experts given means and logvars
        
        Args:
            mus_list (List[torch.Tensor]): list of means of the experts
            log_vars_list (List[torch.Tensor]): list of logvars of the experts
        Returns:
            joint_mu (torch.Tensor): mean of the product of experts
            joint_logvar (torch.Tensor): logvar of the product of experts
        """

        lnT = torch.stack([-l for l in log_vars_list])  # Compute the inverse of variances
        lnV = -torch.logsumexp(lnT, dim=0)  # variances of the product of expert
        mus = torch.stack(mus_list)
        joint_mu = (torch.exp(lnT) * mus).sum(dim=0) * torch.exp(lnV)

        return joint_mu, lnV
    
    
    def kl_divergence_exact(self,mean, prior_mean, log_var, prior_log_var):
        """
        Compute the KL divergence between two gaussians with diagonal covariance matrices.
        
        Parameters:
            mean (torch.Tensor): mean of the first gaussian
            prior_mean (torch.Tensor): mean of the second gaussian
            log_var (torch.Tensor): log variance of the first gaussian
            prior_log_var (torch.Tensor): log variance of the second gaussian
            
        Returns:
            kl (torch.Tensor): the KL divergence between the two gaussians"""
        
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

        return kl.sum()
    
    def subsets(self):
        """
        Returns :
            subsets (list) : all the possible subsets of the modalities.
        """

        subsets = []
        for i in range(1, self.n_modalities+1):
            subsets += combinations(list(self.encoders.keys()), r=i)
        return subsets
    
    
    def subset_encode(self, z_Ls_params, skips, subset):
        """
        Compute all the latent variables and KL divergences for a given subset of modalities.
        
        Args:
            z_Ls_params (Dict[str, ModelOutput]): dictionary containing the mean and logvar of the deepest latent variable
                for each modality.
            skips (Dict[str, List[torch.Tensor]]): dictionary containing the intermediate results of the bottom-up
                layers for each modality.
            subset (List[str]): list of modalities to consider to compute the joint posterior.
            
        Returns:
            z_dict (Dict[str, torch.Tensor]): dictionary containing all the latent variables at each level.
            kl_dict (Dict[str, torch.Tensor]): dictionary containing all the KL divergences at each level."""
        
        
        # Compute the joint posterior q(z_L | x) = p(z_L) * \prod_i q(z_L | x_i )
        list_mus = [z_Ls_params[mod].embedding for mod in subset]
        list_mus.append(torch.zeros_like(list_mus[0])) # add the prior p(z_L) mean = 0
        
        list_log_vars = [z_Ls_params[mod].log_covariance for mod in subset]
        list_log_vars.append(torch.zeros_like(list_log_vars[0])) # add the prior p(z_L) std = 1, logstd = 0
        
        joint_mu, joint_lv = self.poe(list_mus, list_log_vars)
        
        # Sample z_L
        z_l_deepest = dist.Normal(joint_mu, torch.exp(0.5*joint_lv)).rsample()
        
        # Compute KL(q(z_L | x) || p(z_L))
        kl_l_deepest = self.kl_divergence_exact(joint_mu,torch.zeros_like(joint_mu), joint_lv, torch.zeros_like(joint_lv)) # p(z_L) = N(0,1)
        
        # Keep track of all latent variables and KLs
        z_dict = {f'z_{self.n_latent}' : z_l_deepest}
        kl_dict = {f'kl_{self.n_latent}' : kl_l_deepest}
        
        # Sample the rest of the z
        for i in range(self.n_latent-1,0,-1):
            
            h = self.top_down_blocks[i-1](z_dict[f'z_{i+1}'])
            
            # Compute p(z_l|z>l) 
            output = self.prior_blocks[i-1](h) # len(self.prior_blocks) = n_latent - 1
            prior_mu = output.embedding
            prior_log_var = output.log_covariance

            # Compute q(z_l | x, z>l) = p(z_l|z>l) \prod_i q(z_l | x_i, z>l)
            
            list_mus = [prior_mu]
            list_log_vars = [prior_log_var]
            for mod in subset:
                # Compute the parameters of q(z_l | x_i, z>l)
                d = skips[mod][i-1] # skips[mod is of lenght self.n_latent - 1]
                
                concat = torch.cat([h,d], dim=1) # concatenate on the channels 
                
                output = self.get_posterior_block(mod,i-1)(concat)
                list_mus.append(output.embedding)
                list_log_vars.append(output.log_covariance)
            
            joint_mu, joint_lv = self.poe(list_mus, list_log_vars)
            
            # Sample z_l
            z_dict[f'z_{i}'] = dist.Normal(joint_mu, torch.exp(0.5*joint_lv)).rsample()
            
            # Compute KL
            kl_dict[f'kl_{i}'] = self.kl_divergence_exact(joint_mu, prior_mu, joint_lv, prior_log_var)

        return z_dict, kl_dict
    
    def get_posterior_block(self,mod,i):
        """Returns the posterior block for a given modality and level.
        Handles the case where the weights are shared between modalities."""
        if self.share_posterior_weights:
            return self.posterior_blocks[i]
        
        return self.posterior_blocks[mod][i]


    
    
    def loss_subset(self,inputs, z_l_deepest_params,skips, subset):
        """
        
        Compute the negative ELBO loss using a subset of modalities for the posterior.
        
        Args:
            inputs (MultimodalBaseDataset): the input data.
            z_Ls_params (Dict[str, ModelOutput]): dictionary containing the mean and logvar of the deepest latent variable
                for each modality.
            skips (Dict[str, List[torch.Tensor]]): dictionary containing the intermediate results of the bottom-up
                layers for each modality.
            subset (List[str]): list of modalities to consider to compute the joint posterior.
            
        Returns:
            loss (torch.Tensor): the negative ELBO loss.
            kl_dict (Dict[str, torch.Tensor]): dictionary containing all the KL divergences at each level."""
        
        z_dict, kl_dict = self.subset_encode(z_l_deepest_params, skips, subset)
        
        # Reconstruct all modalities using z_1
        recon_loss=0
        for mod in self.decoders:
            output = self.decoders[mod](z_dict['z_1'])
            recon = output.reconstruction
            
            recon_loss += - self.recon_log_probs[mod](recon, inputs.data[mod]).sum()* self.rescale_factors[mod]
        
        # Sum all kls
        kl = 0
        for i in range(1,self.n_latent+1):
            kl += kl_dict[f'kl_{i}']
        
        loss = recon_loss + self.beta * kl
        return loss, kl_dict
    
    
    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """

        Compute the average negative ELBO loss using all possible subsets of modalities for the posterior.

        Args:
            inputs (MultimodalBaseDataset): the input data.
        
        Returns:
            ModelOutput: a ModelOutput instance containing the mean loss and the KL divergences for monitoring.

        """
        
        z_l_deepest_params, skips = self.modality_encode(inputs.data)
        
        subsets = self.subsets()
        
        losses = []
        for subset in subsets:
            loss, kl_dict = self.loss_subset(inputs,z_l_deepest_params,skips,subset)
            losses.append(loss)
        
        loss = torch.stack(losses).mean() # average on all subsets
        
        return ModelOutput(
            loss = loss,
            loss_sum = loss,
            metrics = kl_dict 
        )
            
            
    def encode(self, inputs, cond_mod = "all", N = 1, **kwargs):
        
        """Encode the input data conditioning on the modalities in cond_mod 
        and return the latent variables.
        Args:
            inputs (MultimodalBaseDataset): the input data.
            cond_mod (str, list): the modality to condition on. Either 'all' or a list of modalities.
            N (int): the number of samples to draw from the posterior for each sample.
                Generated latent_variables will have shape (N, n_data, n_latent)
        Returns:
            ModelOutput: a ModelOutput instance containing the latent variables.
        
        """
        
        
        cond_mod = super().encode(inputs,cond_mod,N,**kwargs).cond_mod
        
        z_ls_params, skips = self.modality_encode(inputs.data)
        
        n_data = len(list(z_ls_params.values())[0].embedding)
        
        if N> 1:
            for mod,z_l in z_ls_params.items():
                z_l.embedding = torch.cat([z_l.embedding]*N, dim=0)
                z_l.log_covariance = torch.cat([z_l.log_covariance]*N, dim=0)
                
                skips[mod] = [torch.cat([t]*N, dim=0) for t in skips[mod] ]
        
        z_dict, _ = self.subset_encode(z_ls_params,skips, cond_mod)
        
        flatten = kwargs.pop('flatten', False)
        
        if not flatten and N >1:
            for k in z_dict:
                
                z_dict[k] = z_dict[k].reshape(N,n_data,*z_dict[k].shape[1:])
        
        return ModelOutput(z = z_dict['z_1'], all_z = z_dict, one_latent_space = True)
            
    
    
    def modality_encode(self, data:dict):
        
        """
        Encode each modality on its own. 

        Returns:
            z_Ls_params: a dictionary containing for each modality a ModelOutput instance
                with embedding and logcovariance.
            skips : a dictionary containing a list of tensors for each modality.
        """
        
        # Apply all bottom_up layers, save the intermediate results
        skips = {mod : [] for mod in data}
        z_ls_params = dict()

        for mod in data:
            # Apply first encoder layer
            output = self.encoders[mod](data[mod])
            z = output.embedding
            skips[mod].append(z)
            
            # Apply all intermediate layers
            for i in range(self.n_latent-2):
                z = self.bottom_up_blocks[mod][i](z)
                skips[mod].append(z)
            
            # Apply last layer
            output = self.bottom_up_blocks[mod][-1](z)
            z_ls_params[mod] = output
        
        return z_ls_params, skips
        

    def sanity_check_bottom_up(self, encoders, bottom_up_blocks):
        """Check the coherence of the bottom_up_blocks with the encoders."""
        
        # Check the number of modalities
        if self.n_modalities != len(bottom_up_blocks.keys()):
            raise AttributeError(
                f"The provided number of decoders {len(bottom_up_blocks.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )
        # Check coherence with the encoders keys
        if encoders.keys() != bottom_up_blocks.keys():
            raise AttributeError(
                "The names of the modalities in the encoders dict doesn't match the names of the modalities"
                " in the bottom_up_blocks dict."
            )
        # Check that the number of layers is correct
        for mod in bottom_up_blocks:
            if len(bottom_up_blocks[mod]) != self.n_latent - 1:
                raise AttributeError(f"There must be {self.n_latent - 1} bottom_up_blocks for modality"
                                     f" {mod} but you provided {len(bottom_up_blocks[mod])} layers.")
        # Check that the last layer is an instance of BaseEncoder
            if not isinstance(bottom_up_blocks[mod][-1],BaseEncoder):
                raise AttributeError(f"The last layer in bottom_up_blocks for modality {mod}"
                                     " must be an instance of BaseEncoder")

    def sanity_check_top_down_blocks(self, top_down_blocks):
        """Check the coherence of the top_down_blocks with the model configuration."""
        
        if len(top_down_blocks) != self.n_latent - 1:
            raise AttributeError(f"There must be {self.n_latent-1} modules in top_down_blocks.")
        
    def check_and_set_posterior_blocks(self, posterior_blocks):
        """Check the coherence of the posterior_blocks with the model configuration."""

        # Shared weights : a list of modules was provided
        if isinstance(posterior_blocks,(list,torch.nn.ModuleList)):
            logger.info('Shared weights for the posterior blocks')
            self.share_posterior_weights = True
            if len(posterior_blocks) != self.n_latent - 1:
                raise AttributeError(f"There must be {self.n_latent-1} modules in posterior_blocks[{m}].")
            for block in posterior_blocks:
                if not isinstance(block,BaseEncoder ):
                    raise AttributeError(f'The modules in posterior_blocks[{m}] must be instances of BaseEncoder')
            self.posterior_blocks = torch.nn.ModuleList(posterior_blocks)
            return

        # Not shared weights : a dict of lists of modules was provided
        if isinstance(posterior_blocks,(dict,torch.nn.ModuleDict)):
            logger.info('Not shared weights for the posterior blocks')
            self.share_posterior_weights = False
            assert posterior_blocks.keys() == self.encoders.keys()
            for m, p in posterior_blocks.items():
                if len(p) != self.n_latent - 1:
                    raise AttributeError(f"There must be {self.n_latent-1} modules in posterior_blocks[{m}].")
                for block in p:
                    if not isinstance(block,BaseEncoder ):
                        raise AttributeError(f'The modules in posterior_blocks[{m}] must be instances of BaseEncoder')
            self.posterior_blocks = torch.nn.ModuleDict()
            for mod in posterior_blocks:
                self.posterior_blocks[mod] = torch.nn.ModuleList(posterior_blocks[mod])
            return
        raise AttributeError('posterior_blocks must be a list or a dict')
        
        
            
    def sanity_check_prior_blocks(self, prior_blocks):
        """Check the coherence of the prior_blocks with the model configuration."""
        
        if len(prior_blocks) != self.n_latent - 1:
            raise AttributeError(f"There must be {self.n_latent-1} modules in prior.")
        for block in prior_blocks:
            if not isinstance(block,BaseEncoder ):
                raise AttributeError('The modules in prior_blocks  must be instances of BaseEncoder')

    
    def set_top_down_blocks(self, top_down_blocks):
        """Set the top_down_blocks attribute."""
        self.top_down_blocks = torch.nn.ModuleList(top_down_blocks)


    def set_bottom_up_blocks(self, bottom_up_blocks):
        """Set the bottom_up_blocks attribute."""
        
        self.bottom_up_blocks = torch.nn.ModuleDict()
        for mod in bottom_up_blocks:
            self.bottom_up_blocks[mod] = torch.nn.ModuleList(bottom_up_blocks[mod])
        

    def default_encoders(self, model_config):
        raise ValueError("You have to provide encoders for this model.")
    
    def default_decoders(self, model_config):
        raise ValueError("You have to provide decoders for this model.")