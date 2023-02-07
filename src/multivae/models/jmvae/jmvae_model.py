from ..base import BaseMultiVAE
from .jmvae_config import JMVAEConfig
from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from typing import Tuple, Union
from ..nn.default_architectures import MultipleHeadJointEncoder





class JMVAE(BaseMultiVAE):
    """
    Implements the JMVAE model. 
    
    Args:
        
        model_config (JMVAEConfig): The configuration of the model.
        
        encoders (Dict[BaseEncoder]): A dictionary containing the modalities names and the encoders for each 
            modality (instance of Pythae's BaseEncoder). 

        decoder (Dict[BaseDecoder]): A dictionary containing the modalities names and the encoders for each 
            modality (instance of Pythae's BaseEncoder).
            
        joint_encoder (BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input. 
            If none is provided, one is created from the unimodal encoders. Default : None. 
    """

    def __init__(self, model_config: JMVAEConfig, encoders: dict, decoders: dict, joint_encoder : Union[BaseEncoder, None]=None, **kwargs):
        super().__init__(model_config, encoders, decoders)
        
        if joint_encoder is None:
            # TODO
            joint_encoder = MultipleHeadJointEncoder(self.encoders,model_config)
        
        self.set_joint_encoder(joint_encoder)
        
    def set_joint_encoder(self, joint_encoder):
        "Checks that the provided joint encoder is an instance of BaseEncoder."
        
        if not issubclass(type(joint_encoder), BaseEncoder):
                raise AttributeError(
                    (
                        f"The joint encoder must inherit from BaseEncoder class from "
                        "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                    )
                )
        self.joint_encoder = joint_encoder
        
        