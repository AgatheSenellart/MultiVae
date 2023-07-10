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
            
        if top_decoders is None:
            top_decoders = self.default_top_decoders(model_config)
        
        if joint_encoder is None:
            joint_encoder = self.default_joint_encoder(model_config)
            
        self.set_top_decoders(top_decoders)
        self.set_top_encoders(top_encoders)
        self.joint_encoder(joint_encoder)
        
        self.model_name = 'Nexus'
        
        self.dropout = model_config.dropout_rate
        self.top_level_scalings = model_config.top_level_scalings
        self.beta = model_config.beta
        self.alpha = model_config.alpha
        
        
        
        

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
        
        # Compute the first level elbos
        
        # Compute the second level elbos
        return 