from typing import Tuple, Union, Dict

import torch
import torch.distributions as dist
from torch.nn import ModuleDict
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.normalizing_flows.base import BaseNFConfig, BaseNF
from pythae.models.normalizing_flows.maf import MAFConfig, MAF


from ...data.datasets.base import MultimodalBaseDataset
from ..joint_models import BaseJointModel
from .jnf_config import JNFConfig


class JNF(BaseJointModel):
    
    """
    The JNF model.
    
    Args:
    
        model_config (JNFConfig): An instance of JNFConfig in which any model's parameters is
            made available.

        encoders (Dict[str,BaseEncoder]): A dictionary containing the modalities names and the encoders for each
            modality. Each encoder is an instance of Pythae's BaseEncoder.

        decoders (Dict[str,BaseDecoder]): A dictionary containing the modalities names and the decoders for each
            modality. Each decoder is an instance of Pythae's BaseDecoder.
            
        joint_encoder (BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.
        
        flows (Dict[str,BaseNF]) : A dictionary containing the modalities names and the flows to use for 
            each modality. If None is provided, a default MAF flow is used for each modality.

    
    """

    def __init__(
        self,
        model_config: JNFConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder]  = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        flows: Dict[str, BaseNF]  = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)
        
        if flows is None:
            flows = dict()
            self.config.use_default_flow = True
            for modality in self.encoders:
                flows[modality] = MAF(MAFConfig())
        
        self.set_flows(flows)
                
        self.model_name = "JNF"

        self.warmup = model_config.warmup

    
    def set_flows(flows: Dict[BaseNF]):
        
        # check that the keys corresponds with the encoders keys
        pass