import logging
from typing import Dict, Union
from multivae.models.nn.default_architectures import nn,BaseAEConfig,Encoder_VAE_MLP, Decoder_AE_MLP

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from multivae.models.base import BaseDecoder, BaseEncoder
from pythae.models.normalizing_flows.base import BaseNF
from pythae.models.normalizing_flows.maf import MAF, MAFConfig
from torch.nn import ModuleDict
import torch.distributions as dist


from ...data.datasets.base import MultimodalBaseDataset
from ..base import BaseMultiVAE
from .nexus_config import NexusConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class Nexus(BaseMultiVAE):

    """
    The Nexus model from 
     "Leveraging hierarchy in multimodal generative models for effective cross-modality inference" (Vasco et al 2022)


    Args:

        model_config (NexusConfig): An instance of NexusConfig in which any model's parameters is
            made available.

        encoders (Dict[str, ~multivae.models.base.BaseEncoder]): A dictionary
            containing the modalities names and the encoders for each modality. Each encoder is
            an instance of Multivae's BaseEncoder whose output is of the form:
                ```ModelOutput(
                    embedding = ...,
                    log_covariance = ...,
                )

        decoders (Dict[str, ~multivae.models.base.BaseDecoder]): A dictionary
            containing the modalities names and the decoders for each modality. Each decoder is an
            instance of Pythae's BaseDecoder.

        top_encoders (Dict[str, ~multivae.models.base.BaseEncoder]) : An instance of
            BaseEncoder that takes all the first level representations to generate the messages that will be aggregated.
            Each encoder is
            an instance of Multivae's BaseEncoder whose output is of the form:
                ```ModelOutput(
                    embedding = ...,
                    log_covariance = ...,
                )
            
        joint_encoder (~multivae.models.base.BaseEncoder): The encoder that takes the aggregated message and 
            encode it to obtain the high level latent distribution.
            
        top_decoders (Dict[str, ~multivae.models.base.BaseDecoder]) : Top level decoders from the joint representation
            to the modalities specific representations.

    """

    def __init__(
        self,
        model_config: NexusConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        top_encoders: Dict[str, BaseEncoder] = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        top_decoders: Dict[str, BaseNF] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, **kwargs)
        
        if top_encoders is None:
            top_encoders = self.default_top_encoders(model_config)
        else:
            self.model_config.custom_architectures.append('top_encoders')
            
        if top_decoders is None:
            top_decoders = self.default_top_decoders(model_config)
        else:
            self.model_config.custom_architectures.append('top_decoders')
        
        if joint_encoder is None:
            joint_encoder = self.default_joint_encoder(model_config)
        else:
            self.model_config.custom_architectures.append('joint_encoder')
            
        self.set_top_decoders(top_decoders)
        self.set_top_encoders(top_encoders)
        self.joint_encoder(joint_encoder)
        
        self.model_name = 'Nexus'
        
        self.dropout = model_config.dropout_rate
        self.top_level_scalings = model_config.top_level_scalings
        self.set_bottom_betas(model_config.bottom_betas)
        self.set_gammas(model_config.gammas)
        
        self.beta = model_config.beta
        self.aggregator_function = model_config.aggregator
        
        
        
    
    def set_bottom_betas(self, bottom_betas):
        if bottom_betas is None:
            self.bottom_betas =  {m : 1. for m in self.encoders}
        else :
            if bottom_betas.keys() != self.encoders.keys():
                raise AttributeError('The bottom_betas keys do not match the modalities'
                                     'names in encoders.')
            else:
                self.bottom_betas =  bottom_betas
                
    def set_gammas(self, gammas):
        if gammas is None:
            self.gammas =  {m : 1. for m in self.encoders}
        else :
            if gammas.keys() != self.encoders.keys():
                raise AttributeError('The gammas keys do not match the modalities'
                                     'names in encoders.')
            else:
                self.gammas =  gammas
        

    def default_encoders(self, model_config : NexusConfig):
        
        if model_config.input_dims is None or model_config.modalities_specific_dim is None:
            raise AttributeError("Please provide encoders architectures or "
                                 "valid input_dims and modalities_specific_dim in the"
                                 "model configuration")
        
        encoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(input_dim=model_config.input_dims[mod], 
                                 latent_dim=model_config.modalities_specific_dim[mod])
            encoders[mod] = Encoder_VAE_MLP(config)
        return encoders
    
    def default_decoders(self, model_config: NexusConfig):
        
        if model_config.input_dims is None or model_config.modalities_specific_dim is None:
            raise AttributeError("Please provide encoders architectures or "
                                 "valid input_dims and modalities_specific_dim in the"
                                 "model configuration")
        
        decoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(input_dim=model_config.input_dims[mod], 
                                 latent_dim=model_config.modalities_specific_dim[mod])
            decoders[mod] = Decoder_AE_MLP(config)
        return decoders
    
    def default_top_encoders(self, model_config: NexusConfig):
        
        if model_config.modalities_specific_dim is None:
            raise AttributeError("Please provide encoders architectures or "
                                 "valid input_dims and modalities_specific_dim in the"
                                 "model configuration")
        
        encoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(input_dim=model_config.modalities_specific_dim[mod], 
                                 latent_dim=model_config.msg_dim)
            encoders[mod] = Encoder_VAE_MLP(config)
        return encoders
    
    def default_top_decoders(self, model_config: NexusConfig):
        
        if model_config.input_dims is None or model_config.modalities_specific_dim is None:
            raise AttributeError("Please provide encoders architectures or "
                                 "valid input_dims and modalities_specific_dim in the"
                                 "model configuration")
        
        decoders = nn.ModuleDict()
        for mod in model_config.input_dims:
            config = BaseAEConfig(input_dim=model_config.modalities_specific_dim[mod], 
                                 latent_dim=model_config.msg_dim)
            decoders[mod] = Encoder_VAE_MLP(config)
        return decoders
    
    def default_joint_encoder(self, model_config: NexusConfig):
        
        return Encoder_VAE_MLP(
            BaseAEConfig(input_dim=(model_config.msg_dim,), latent_dim=model_config.latent_dim)
        )
        
        
    def set_top_encoders(self,encoders):
        
        self.top_encoders = nn.ModuleDict()
        for k in encoders:
            if not isinstance(encoders[k],BaseEncoder):
                raise AttributeError("Top Encoders must be instances of multivae.models.base.BaseEncoder")
            else :
                self.top_encoders[k] = encoders[k]
                
    def set_top_decoders(self,decoders):
        
        self.top_decoders = nn.ModuleDict()
        for k in decoders:
            if not isinstance(decoders[k],BaseDecoder):
                raise AttributeError("Top Decoders must be instances of multivae.models.base.BaseDecoder")
            else :
                self.top_decoders[k] = decoders[k]
                
    def set_joint_encoder(self,joint_encoder):
        
        if not isinstance(joint_encoder,BaseEncoder):
                raise AttributeError("Joint encoder must be an instance of multivae.models.base.BaseEncoder")
        else :
            self.joint_encoder = joint_encoder
                
        
        
    def forward(self, inputs: MultimodalBaseDataset):
        
        # Compute the first level representations and ELBOs
        modalities_msg = dict()
        first_level_elbos = 0
        first_level_z = dict()
        
        for m in inputs.data:
            
            output_m = self.encoders[m](inputs.data[m])
            mu, logvar = output_m.embedding, output_m.log_covariance
            sigma = torch.exp(0.5*logvar)
            
            z = dist.Normal(mu,sigma).rsample()
            
            # re-decode
            recon = self.decoders[m](z).reconstruction
            
             
            # Compute the ELBO
            logprob = -(self.recon_log_probs[m](recon, inputs.data[m])
                        * self.rescale_factors[m]).reshape(recon.size(0), -1).sum(-1)
            
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            elbo = logprob + KLD*self.bottom_betas[m]
            
            if hasattr(inputs, 'masks'):
                elbo = elbo*inputs.masks[m].float()
                z = z*inputs.masks[m].float()
            
            first_level_elbos += elbo
            
            msg = self.top_encoders[m](z).embedding
            modalities_msg[m] = msg
            first_level_z[m] = z
            
        # Compute the aggregation
        if self.aggregator_function == 'mean':
            aggregated_msg = torch.sum(
                torch.stack(list(modalities_msg.values()), dim=0),dim=0
            )
            
            if hasattr(inputs,'masks'):
                normalization_per_sample = torch.stack(
                    [inputs.masks[m] for m in inputs.masks], dim=0).sum(0)
            
            else:
                normalization_per_sample = self.n_modalities
            
            aggregated_msg /= normalization_per_sample
            
        else:
            raise AttributeError(f'The aggregator function {self.aggregator}'
                                 'is not supported at the moment for the nexus model.')
        
        # Compute the higher level latent variable and ELBO
        joint_output = self.joint_encoder(aggregated_msg)
        joint_mu, joint_log_var = joint_output.mu, joint_output.log_covariance
        joint_sigma = torch.exp(0.5*joint_log_var)
        
        joint_z = dist.Normal(joint_mu, joint_sigma).rsample()
        
        joint_elbo = 0
        for m in self.top_decoders:
            
            recon = self.top_decoders[m](joint_z).reconstruction
            
            joint_elbo += -(dist.Normal(recon,scale=1).log_prob(first_level_z[m])
                        * self.gammas[m]).sum(-1)
        
        joint_KLD = -0.5 * torch.sum(1 + joint_log_var - joint_mu.pow(2) - joint_log_var.exp(), dim=1)
        joint_elbo += self.beta*joint_KLD
        
        total_loss = joint_elbo + first_level_elbos
        
            
        
        return ModelOutput(loss = total_loss.mean(0), metrics={})