from typing import Tuple, Union, Dict

import numpy as np 
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
        
        self.use_likelihood_rescaling = model_config.use_likelihood_rescaling
        if self.use_likelihood_rescaling:
            if self.input_dims is None:
                raise AttributeError(" inputs_dim = None but (use_likelihood_rescaling = True"
                                     " in model_config)"
                                     " To compute likelihood rescalings we need the input dimensions."
                                     " Please provide a valid dictionary for input_dims.")
            else :
                self.rescale_factors = {k: 1/np.prod(self.input_dims[k]) for k in self.input_dims}
        else:
            self.rescale_factors =  {k: 1 for k in self.encoders} 
                # above, we take the modalities keys in self.encoders as input_dims may be None
                
        self.model_name = "JNF"
        self.warmup = model_config.warmup

    
    def set_flows(self,flows: Dict[BaseNF]):
        
        # check that the keys corresponds with the encoders keys
        if flows.keys() != self.encoders.keys():
            raise AttributeError(f"The keys of provided flows : {list(flows.keys())}"
                                 f" doesn't match the keys provided in encoders {list(self.encoders.keys())}"
                                 " or input_dims.")
        
        # Check that the flows are instances of BaseNF
        self.flows = ModuleDict()
        for m in flows:
            if isinstance(flows[m],BaseNF):
                self.flows[m] = flows[m]
            else:
                raise AttributeError("The provided flows must be instances of the Pythae's BaseNF "
                                     " class.")
        return 
    
    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        
        epoch = kwargs.pop("epoch", 1)
        

        # First compute the joint ELBO
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)
        z_joint = qz_xy.rsample()

        recon_loss = 0

        # Decode in each modality
        len_batch = 0
        for mod in self.decoders:
            x_mod = inputs.data[mod]
            len_batch = len(x_mod)
            recon_mod = self.decoders[mod](z_joint).reconstruction
            recon_loss += (-0.5 * torch.sum((x_mod - recon_mod) ** 2))*self.rescale_factors[m]

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        if epoch < self.warmup:
            
            return ModelOutput(
                recon_loss = recon_loss/len_batch,
                KLD = KLD/len_batch,
                loss = -(recon_loss+KLD)/len_batch
            )
            
        else :
            ljm = 0
            for m in self.encoders:
                
                mod_output = self.encoders[mod](inputs.data[mod])
                mu0, log_var0 = mod_output.embedding, mod_output.log_covariance

                sigma0 = torch.exp(0.5 * log_var0)
                qz_x0= dist.Normal(mu0, sigma0)
                
                # Compute -ln q_\phi_mod(z_joint|x_mod)
                z_joint = z_joint.detach() # no backpropagation on z_joint
                flow_output = self.flows[mod](z_joint)
                ljm +=  - (qz_x0.log_prob(z_joint) + flow_output.log_abs_det_jac).sum()

            
            return ModelOutput(
                recon_loss = recon_loss/len_batch,
                KLD = KLD/len_batch,
                loss = ljm/len_batch
            )

    def encode(self, inputs: MultimodalBaseDataset, cond_mod: Union[list, str] = "all", N: int = 1, **kwargs) -> ModelOutput:
        return super().encode(inputs, cond_mod, N, **kwargs)
    
    
        