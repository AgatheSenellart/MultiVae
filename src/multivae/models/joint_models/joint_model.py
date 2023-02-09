from ..base import BaseMultiVAE
from .joint_model_config import BaseJointModelConfig
from pythae.models.nn.base_architectures import BaseEncoder, BaseDecoder
from typing import Tuple, Union
from ..nn.default_architectures import MultipleHeadJointEncoder





class BaseJointModel(BaseMultiVAE):
    """
    Base Class for models using a joint encoder.
    
    Args:
        
        model_config (BaseJointModelConfig): The configuration of the model.
        
        encoders (Dict[BaseEncoder]): A dictionary containing the modalities names and the encoders for each 
            modality. Each encoder is an instance of Pythae's BaseEncoder class.

        decoder (Dict[BaseDecoder]): A dictionary containing the modalities names and the decoders for each 
            modality. Each decoder is an instance of Pythae's BaseDecoder class. 
            
        joint_encoder (BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input. 
            If none is provided, one is created from the unimodal encoders. Default : None. 
    """

    def __init__(self, model_config: BaseJointModelConfig, encoders: dict, decoders: dict, joint_encoder : Union[BaseEncoder, None]=None, **kwargs):
        super().__init__(model_config, encoders, decoders)
        
        if joint_encoder is None:
            # Create a MultiHead Joint Encoder MLP
            joint_encoder = MultipleHeadJointEncoder(self.encoders,model_config)
            model_config.use_default_joint = True
        
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
        
        