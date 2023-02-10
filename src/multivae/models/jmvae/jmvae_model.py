from ..joint_models import BaseJointModel
from .jmvae_config import JMVAEConfig
from typing import Tuple, Union
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from ...data.datasets.base import MultimodalBaseDataset
from pythae.models.base.base_utils import ModelOutput
import torch.distributions as dist
import torch

class JMVAE(BaseJointModel):
    
    """The JMVAE model from the paper 'Joint Multimodal Learning with Deep Generative Models' 
    (Suzuki et al, 2016), http://arxiv.org/abs/1611.01891."""
    
    def __init__(self, model_config: JMVAEConfig, encoders: dict=None, decoders: dict=None, joint_encoder: Union[BaseEncoder, None] = None, **kwargs):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        self.model_name = 'JMVAE'

        self.alpha = model_config.alpha
        self.warmup = model_config.warmup


    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Performs a forward pass of the JMVAE model on inputs.

        Args:
            inputs (MultimodalBaseDataset)
            warmup (int) : number of warmup epochs to do. The weigth of the regularization augments linearly to reach 1 at the end of
                the warmup. The enforces the optimization of the reconstruction term only at first. 
            epoch (int) : the epoch number during which forward is called.    

        Returns:
            ModelOutput
        """

        epoch = kwargs.pop('epoch', 1)

        # Compute the reconstruction term
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        sigma = torch.exp(0.5*log_var)
        qz_xy = dist.Normal(mu, sigma)
        z_joint = qz_xy.rsample()
        
        recon_loss = 0
        
        # Decode in each modality
        for mod in self.decoders:
            x_mod = inputs.data[mod]
            recon_mod = self.decoders[mod](z_joint).reconstruction
            recon_loss += -0.5*torch.sum((x_mod - recon_mod)**2)
            
        
        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())        
        
        # Compute the KL between unimodal and joint encoders
        LJM = 0
        for mod in self.encoders:
            output = self.encoders[mod](inputs.data[mod])
            uni_mu, uni_log_var = output.embedding, output.log_covariance
            LJM += 1/2*(uni_log_var-log_var +  (torch.exp(log_var) + (mu - uni_mu)**2)/torch.exp(uni_log_var) - 1)
        
        LJM = LJM.sum()*self.alpha
        
        # Compute the total loss to minimize
        
        reg_loss = KLD + LJM
        beta = min(1, epoch / self.warmup)
        loss = recon_loss - beta*reg_loss
        
        output = ModelOutput(loss=loss)
        
        return output