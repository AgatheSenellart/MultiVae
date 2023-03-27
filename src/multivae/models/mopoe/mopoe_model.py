from itertools import chain, combinations
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from numpy.random import choice
from pythae.models.base.base_utils import ModelOutput
from scipy.special import comb
from torch.distributions import kl_divergence

from multivae.data.datasets.base import IncompleteDataset, MultimodalBaseDataset

from ..base import BaseMultiVAE
from .mopoe_config import MoPoEConfig


class MoPoE(BaseMultiVAE):
    
    """ Implementation for the Mixture of Product of experts model from 
    'Generalized Multimodal ELBO' Sutter 2021 (https://arxiv.org/abs/2105.02470)
    
    This implementation is heavily based on the official one at 
    https://github.com/thomassutter/MoPoE
    

    """
    
    def __init__(self, model_config: MoPoEConfig, encoders: dict = None, decoders: dict = None):
        super().__init__(model_config, encoders, decoders)
        
        self.beta = model_config.beta
        self.model_name = 'MoPoE'
        list_subsets = self.model_config.subsets
        if list_subsets is None:
            list_subsets = self.all_subsets()
        elif type(list_subsets) == dict:
            list_subsets = list(self.model_config.subsets.values())
        self.set_subsets(list_subsets)
        self.decoder_scale = model_config.decoder_scale
        
    def all_subsets(self):
        """
        Returns a list containing all possible subsets of the modalities. 
        """
        xs = list(self.encoders.keys())
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                          range(len(xs)+1))
        return subsets_list
    
    def set_subsets(self,subsets_list ):
        subsets = dict();
        for k, mod_names in enumerate(subsets_list):
            mods = [];
            for l, mod_name in enumerate(sorted(mod_names)):
                if mod_name not in self.encoders.keys() :
                    raise AttributeError(f'The provided subsets list contains unknown modality name {mod_name}.'
                                         ' that is not the encoders dictionary or inputs_dim dictionary.')
                mods.append(mod_name)
            key = '_'.join(sorted(mod_names));
            subsets[key] = mods;
        self.subsets = subsets
        self.model_config.subsets = subsets
        return 
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        z = dist.Normal(mu, std).rsample()
        return z
    
    def calc_joint_divergence(self, mus : torch.Tensor, logvars:torch.Tensor, weights:torch.Tensor, normalization : int =None):
        """Computes the KL divergence between the mixture of experts and the prior, by 
        developping into the sum of the tractable KLs divergences of each expert.

        Args:
            mus (Tensor): The means of the experts.
            logvars (Tensor): The logvars of the experts.
            weights (Tensor): The weights of the experts.
            normalization (int, optional): If an int is provided, the kld is summed over sampled and normalized by 
                the value of normalization. Defaults to None.

        Returns:
            Tensor, Tensor: The group divergence summed over modalities, A tensor containing the KL terms for each experts.
        """
        weights = weights.clone();
        weights = weights / weights.sum()
        
        num_mods = mus.shape[0];
        num_samples = mus.shape[1];
        if normalization is not None:
            klds = torch.zeros(num_mods);
        else:
            klds = torch.zeros(num_mods, num_samples);
        device = mus.device
        klds = klds.to(device)
        weights = weights.to(device)
        for k in range(0, num_mods):
            kld_ind = -0.5 * torch.sum(1 - logvars[k,:,:].exp() - mus[k,:,:].pow(2) + logvars[k,:,:])
            if normalization is not None:
                kld_ind /= normalization
            
            if normalization is not None:
                klds[k] = kld_ind;
            else:
                klds[k,:] = kld_ind;
        if normalization is None:
            weights = weights.unsqueeze(1).repeat(1, num_samples);
        group_div = (weights*klds).sum(dim=0);
        
        divs = dict();
        divs['joint_divergence'] = group_div; divs['individual_divs'] = klds
        return divs
        
    # def calc_joint_divergence(self, mus, logvars, weights):
    #     """Compute the KL divergence between the joint distribution and the prior 
    #     that is considered a static Normal distribution here. """
        

    #     weights = weights.clone();
    #     weights = weights / weights.sum()
    #     div_measures = self.calc_group_divergence_moe(
    #                                              mus,
    #                                              logvars,
    #                                              weights,
    #                                              normalization=len(mus[0]));
    #     divs = dict();
    #     divs['joint_divergence'] = div_measures[0]; divs['individual_divs'] = div_measures[1]
    #     return divs;
    
    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        
        # Compute latents parameters for all subsets
        latents = self.inference(inputs)
        results = dict()
        # results['latents'] = latents
        # results['group_distr'] = latents['joint']
        
        # Compute the divergence to the prior
        shared_embeddings = self.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);
        ndata = len(shared_embeddings)
        div = self.calc_joint_divergence(latents['mus'],
                                        latents['logvars'],
                                        latents['weights'],
                                        normalization = ndata) 
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        # Compute the reconstruction losses for each modality
        loss = 0
        for m, m_key in enumerate(self.encoders.keys()):
            # reconstruct this modality from the shared embeddings representation
            recon = self.decoders[m_key](shared_embeddings).reconstruction
            m_rec = self.recon_losses[m_key](recon, inputs.data[m_key])* self.rescale_factors[m_key]/self.decoder_scale
            
            # m_s_mu, m_s_logvar = enc_mods[m_key + '_style'];
            # if self.flags.factorized_representation:
            #     m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar);
            # else:
            #     m_s_embeddings = None;
            # m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings));
            results['recon_' +  m_key] = m_rec.sum()/ndata
            loss += m_rec.sum()/ndata
        loss = loss + self.beta * results['joint_divergence']

        return ModelOutput(loss=loss,metrics = results)

    def modality_encode(self, inputs: Union[MultimodalBaseDataset,IncompleteDataset], **kwargs):
        """Computes for each modality, the parameters mu and logvar of the 
        unimodal posterior.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.

        Returns:
            dict: Containing for each modality the encoder output.
        """
        encoders_outputs = dict();
        for m, m_key in enumerate(self.encoders.keys()):

            input_modality = inputs.data[m_key]
            output = self.encoders[m_key](input_modality)
            encoders_outputs[m_key] = output
            # latents[m_key + '_style'] = l[:2]
            # latents[m_key] = l[2:]

        return encoders_outputs
    
    def poe(self,mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar
    
    def poe_fusion(self, mus : torch.Tensor, logvars : torch.Tensor, weights=None):
        
        # Following the original implementation : add the prior when we consider the 
        # subset that contains all the modalities
        if mus.shape[0] == len(self.encoders.keys()):
            
            num_samples = mus[0].shape[0]
            device = mus.device
            mus = torch.cat((mus, torch.zeros(1, num_samples,
                             self.latent_dim).to(device)),
                            dim=0)
            logvars = torch.cat((logvars, torch.zeros(1, num_samples,
                                 self.latent_dim).to(device)),
                                dim=0)
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_poe, logvar_poe = self.poe(mus, logvars)
        return [mu_poe, logvar_poe]
    
    def _filter_inputs_with_masks(
        self, inputs: IncompleteDataset, subset: Union[list, tuple]
    ):
        """Returns a filtered dataset containing only the samples that are available
        in all the modalities contained in subset."""

        filter = torch.tensor(
            True,
        ).to(inputs.masks[subset[0]].device)
        for mod in subset:
            filter = torch.logical_and(filter, inputs.masks[mod])

        filtered_inputs = MultimodalBaseDataset(
            data = {k : inputs.data[k][filter] for k in inputs.data},
        )
        return filtered_inputs, filter



    def inference(self, inputs: MultimodalBaseDataset, **kwargs):
        """
        
        Args:
            inputs (MultimodalBaseDataset): The data. 

        Returns:
            _type_: _description_
        """
        
        latents = dict()
        enc_mods = self.modality_encode(inputs)
        latents['modalities'] = enc_mods
        device = inputs.data[list(inputs.data.keys())[0]].device
        mus = torch.Tensor().to(device)
        logvars = torch.Tensor().to(device)
        distr_subsets = dict()
        for k, s_key in enumerate(self.subsets.keys()):
            if s_key != '':

                mods = self.subsets[s_key]
                mus_subset = torch.Tensor().to(device)
                logvars_subset = torch.Tensor().to(device)
                
                mods_avail = True
                # # For a complete dataset, build a filter with masks equal to True 
                # # for every samples
                # if len(inputs.data[mods[0]].shape)==1: # only one sample 
                #     filter = torch.tensor(True)
                # else :
                #     filter = torch.ones(len(inputs.data[mods[0]])).bool()
                if hasattr(inputs, 'masks'):
                    filtered_inputs, filter = self._filter_inputs_with_masks(inputs,mods)
                    mods_avail = torch.all(filter) 
                if mods_avail :
                    for m, mod in enumerate(mods):
                        mus_mod = enc_mods[mod].embedding
                        log_vars_mod = enc_mods[mod].log_covariance

                        mus_subset = torch.cat((mus_subset,
                                                mus_mod.unsqueeze(0)),
                                            dim=0)

                        

                        logvars_subset = torch.cat((logvars_subset,
                                                    log_vars_mod.unsqueeze(0)),
                                                dim=0);
                        
                    # Compute the subset posterior parameters : not used in PoE fusion
                    # weights_subset = ((1/float(len(mus_subset)))*
                    #                   torch.ones(len(mus_subset)).to(device))
                    
                    # Case with only one sample : adapt the shape
                    if len(mus_subset.shape)==2:
                        mus_subset = mus_subset.unsqueeze(1)
                        logvars_subset = logvars_subset.unsqueeze(1)
                    
                    s_mu, s_logvar = self.poe_fusion(mus_subset,
                                                          logvars_subset)
                    

                    
                    distr_subsets[s_key] = [s_mu, s_logvar]
                    
                    # Add the subset posterior to be part of the mixture of experts
                    mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                    logvars = torch.cat((logvars, s_logvar.unsqueeze(0)),
                                            dim=0)
       
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(device);
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights);
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        latents['mus'] = mus
        latents['logvars'] = logvars
        latents['weights'] = weights
        latents['joint'] = [joint_mu, joint_logvar]
        latents['subsets'] = distr_subsets
        return latents
    
    def encode(self, inputs: MultimodalBaseDataset, cond_mod: Union[list, str] = "all", N: int = 1, **kwargs) -> ModelOutput:
        
        # TODO : deal with the case where you want to encode
        # an incomplete dataset
        
        # If the input cond_mod is a string : convert it to a list
        if type(cond_mod) == str:
            if cond_mod == "all":
                cond_mod = list(self.encoders.keys())
            elif cond_mod in self.encoders.keys():
                cond_mod = [cond_mod]
            else:
                raise AttributeError(
                    'If cond_mod is a string, it must either be "all" or a modality name'
                    f" The provided string {cond_mod} is neither."
                )
                
        # Compute the str associated to the subset
        key = '_'.join(sorted(cond_mod))
        
        # If the dataset is incomplete, keep only the samples availables in all cond_mod 
        # modalities
        if isinstance(inputs, IncompleteDataset):
            inputs, filter = self._filter_inputs_with_masks(inputs,cond_mod)
            
        
        latents_subsets = self.inference(inputs)
        mu, log_var = latents_subsets['subsets'][key]
        sample_shape = [N] if N>1 else []
        z = dist.Normal(mu, torch.exp(0.5*log_var)).rsample(sample_shape)
        flatten = kwargs.pop("flatten", False)
        if flatten:
            z = z.reshape(-1, self.latent_dim)

        return ModelOutput(z=z, one_latent_space=True)
        
    
    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = torch.ones((mus.shape[0],)).to(mus.device)
        weights = weights / weights.sum()
        #mus = torch.cat(mus, dim=0);
        #logvars = torch.cat(logvars, dim=0);
        mu_moe, logvar_moe = self.mixture_component_selection(
                                                                mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];

    def mixture_component_selection(self, mus, logvars, w_modalities):
        #if not defined, take pre-defined weights
        num_components = mus.shape[0];
        num_samples = mus.shape[1];
        
        idx_start = [];
        idx_end = []
        for k in range(0, num_components):
            if k == 0:
                i_start = 0;
            else:
                i_start = int(idx_end[k-1]);
            if k == w_modalities.shape[0]-1:
                i_end = num_samples;
            else:
                i_end = i_start + int(torch.floor(num_samples*w_modalities[k]));
            idx_start.append(i_start);
            idx_end.append(i_end);
        idx_end[-1] = num_samples;
        mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
        logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
        return [mu_sel, logvar_sel];

    def compute_joint_nll(
        self,
        inputs: Union[MultimodalBaseDataset, IncompleteDataset],
        K: int = 1000,
        batch_size_K: int = 100,
    ):
        # Only keep the complete samples
        all_modalities = list(self.encoders.keys())
        if hasattr(inputs, "masks"):
            filtered_inputs, filter = self._filter_inputs_with_masks(
                inputs, all_modalities
            )
            
        else:
            filtered_inputs = inputs

        # Compute the parameters of the joint posterior
        mu, log_var = self.inference(filtered_inputs)['joint']

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

                    dim_reduce = tuple(range(1, len(recon.shape)))
                    lpx_zs += self.recon_log_probs[mod](recon, x_m).sum(dim=dim_reduce)

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

        return -ll / n_data
