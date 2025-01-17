from pydantic.dataclasses import dataclass
from typing import Literal, Dict, List
from pythae.config import BaseConfig
from dataclasses import field

@dataclass
class CVAEConfig(BaseConfig):
    """This is the configuration class for the Conditional Variational Autoencoder model.

    Args:
        
        input_dims (dict[str,tuple]) : The modalities'names (str) and input shapes (tuple).
        latent_dim (int): The dimension of the latent space. Default: 10.
        conditioning_modality (str): The modality to condition the model on.
        main_modality (str): The main modality to reconstruct.
        decoder_dist (str): The decoder distribution to use. Possible values ['normal','bernoulli','laplace', 'categorical'].
            For Bernoulli distribution, the decoder is expected to output **logits**.
        decoder_dist_params (dict) : To eventually specify parameters for the output decoder distribution. 
            Default to None.
        
    """
    
    
    conditioning_modality:str
    main_modality:str
    input_dims:dict = None
    latent_dim:int = 10
    decoder_dist:Literal['normal', 'laplace','bernoulli','categorical']='normal'
    decoder_dist_params:dict=field(default_factory=lambda: {})
    custom_architectures:list = field(default_factory=lambda: [])
