from typing import Dict, Union
import torch
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder
from multivae.models.nn import BaseJointEncoder
from torch.nn import Module, ModuleDict

from ...data.datasets import MultimodalBaseDataset
from .gmc_config import GMCConfig

from multivae.models.base import BaseModel

class GMC(BaseModel):
    def __init__(self, model_config: GMCConfig, processors: Dict[str, BaseEncoder], shared_encoder : BaseEncoder,joint_encoder : BaseJointEncoder=None, ) -> None:
        # """
        # Implements the Geometric Multimodal Contrastive Learning from :
        # https://arxiv.org/abs/2202.03390
        
        # This implementation is based on the original implementation : https://github.com/miguelsvasco/gmc
        
        # Args:
        #     config (GMCConfig) : Contains all the hyperparameters for the model
        #     processors (Dict[str, BaseEncoder]) : A dictionary containing an encoder for each modality. Each encoder is
        #         expected to be a pythae BaseEncoder instance.
        #     joint_encoder (BaseJointEncoder) : The joint encoder used for computing the joint representation
        #     shared_encoder (BaseEncoder) : The shared projection head. 
        # """
        
        super().__init__(model_config)

        
        self.model_name = 'GMC'
        self.n_modalities = model_config.n_modalities
        self.latent_dim = model_config.latent_dim
        self.common_dim = model_config.common_dim
        self.temperature = model_config.temperature
        self.set_processors(processors)
        self.set_shared_encoder(shared_encoder)
        self.model_config.custom_architectures.extend([
            "processors", 
            "shared_encoder"
        ])
        
        if model_config.loss == "between_modality_pairs" or joint_encoder is None:
            self.loss = "pairs"

        else :
            self.set_joint_encoder(joint_encoder)
            self.model_config.custom_architectures.append("joint_encoder")
            self.loss = "joint"
        


    def set_processors(self, processors):
        self.processors = ModuleDict()
        assert (
            len(processors) == self.n_modalities
        ), "The number of provided processors doesn't match the number of modalities."

        for m in processors:
            if not isinstance(processors[m], BaseEncoder):
                raise AttributeError(
                    "The GMC processors must be instances of pythae BaseEncoder class."
                )
            if processors[m].latent_dim != self.common_dim:
                raise AttributeError(
                    f"One of the GMC processor network (modality : {m}) doesn't have the same common dim as the model"
                    f" itself. ({processors[m].latent_dim} is different {self.common_dim})"
                )
            self.processors[m] = processors[m]
            
    def set_shared_encoder(self, shared_encoder):
        if not isinstance(shared_encoder, BaseEncoder):
            raise AttributeError("The shared encoder must be an instance of pythae BaseEncoder class")
        if shared_encoder.latent_dim != self.latent_dim :
            raise AttributeError("The shared encoder must have the same latent dim as the model. "
                                 f"In the model config latent dim is {self.latent_dim} different than shared encoder latent dim {shared_encoder.latent_dim}")
        self.shared_encoder = shared_encoder
        
    def set_joint_encoder(self, joint_encoder):
        
        if not isinstance(joint_encoder, BaseJointEncoder):
            raise AttributeError("The joint encoder must be an instance of the ~multivae.models.nn.base_architectures.BaseJointEncoder class.")
        if joint_encoder.latent_dim != self.common_dim:
            raise AttributeError("The joint encoder 'latent_dim' attribute doesn't match the model config 'common_dim' attribute. ")
        self.joint_encoder = joint_encoder
        
    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Basic training step"""
        
        # Compute all the modalities specific representations
        
        modalities_z = dict()
        len_batch = 0
        for m in inputs.data:
            h = self.processors[m](inputs.data[m]).embedding
            z = self.shared_encoder(h).embedding
            modalities_z[m] = z
        len_batch = z.shape[0]
        
        # Compute the joint representation
        if self.loss == 'joint':
            joint_h = self.joint_encoder(inputs.data).embedding
            joint_z = self.shared_encoder(joint_h).embedding
        
            # Compute the loss
            mean_loss = self.infonce(modalities_z, joint_z)
            output = ModelOutput(loss = mean_loss,
                                 loss_sum = mean_loss*len_batch, metrics = {})
        
        elif self.loss == 'pairs':
            mean_loss = self.infonce_pairs(modalities_z)
            output = ModelOutput(loss=mean_loss, loss_sum = mean_loss*len_batch, metrics={})
        
        else :
            raise NotImplementedError()
        
        return output
    
    def infonce_pairs(self, modalities_z):
        """A variant of the GMC objective where no joint representation is computed and the 
        positive pairs are pairs of different modalities for the same sample. 

        Args:
            modalities_z (Dict[str, Tensor]):contains the batch embeddings for each modality 
        """
        modalities_z = list(modalities_z.values())
        batch_size = len(modalities_z[0])
        joint_mod_loss_sum = 0
        
        for i in range(len(modalities_z)):
            for j in range(i+1, len(modalities_z)):
                
                both_mod = torch.cat(
                    [modalities_z[i], modalities_z[j]], dim=0
                )
                # [2*B, 2*B]
                sim_matrix_both_mod = torch.exp(
                    torch.mm(both_mod, both_mod.t().contiguous()) / self.temperature
                )
                # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
                mask_both_mod = (
                    torch.ones_like(sim_matrix_both_mod)
                    - torch.eye(2 * batch_size, device=sim_matrix_both_mod.device)
                ).bool()
                # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
                sim_matrix_both_mod = sim_matrix_both_mod.masked_select(
                    mask_both_mod
                ).view(2 * batch_size, -1)

                # Positive pairs: cosine loss joint-modality
                pos_sim_both_mod = torch.exp(
                    torch.sum(
                        modalities_z[i] * modalities_z[j], dim=-1
                    )
                    / self.temperature
                )
                # [2*B]
                pos_sim_both_mod = torch.cat([pos_sim_both_mod, pos_sim_both_mod], dim=0)
                loss_both_mod = -torch.log(
                    pos_sim_both_mod / sim_matrix_both_mod.sum(dim=-1)
                )
                joint_mod_loss_sum += loss_both_mod

        loss = torch.mean(joint_mod_loss_sum)
        
        return loss
        
    
    def infonce(self, modalities_z, joint_z):
        
        """
        The objective used in the paper of the GMC model, where positive pairs are the pairs with one modality and 
        the joint representation.

        Returns:
            float: the loss for the batch
        """
        
        batch_size = len(joint_z)
        joint_mod_loss_sum = 0
        
        for mod in modalities_z:
            # Negative pairs : joint and mod
            out_joint_mod = torch.cat(
                [joint_z, modalities_z[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / self.temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                torch.ones_like(sim_matrix_joint_mod)
                - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    joint_z * modalities_z[mod], dim=-1
                )
                / self.temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        
        return loss
        
    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: str = "all",
        **kwargs,
    ) :
        """Encode the data using the one or several modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.
            cond_mod (str, optional): Either a modality's name or 'all' (to compute joint encoding). Defaults to "all".

        Raises:
            AttributeError: _description_
        """
        
        
        
        # joint encoding
        if cond_mod == 'all':
            if self.loss == 'pairs':
                raise AttributeError("The encode function was called with cond_mod = 'all' but ",
                                     "this model doesn't have a trained joint encoder since the loss ",
                                     'in model_config is "between_modalities_pairs". ',
                                     "Argument cond_mod must be a modality name : ",
                                     f"for instance one of {self.processors.keys()}.")
            h = self.joint_encoder(inputs.data).embedding
            output = self.shared_encoder(h)
        # modality encoding
        elif cond_mod in self.processors:
            # output = self.shared_encoder(self.processors[cond_mod](inputs.data[cond_mod]).embedding)
            output = self.processors[cond_mod](inputs.data[cond_mod])

        else :
            raise AttributeError("cond_mod must be : either a modality's name or equal to 'all' (for joint representation). ")
        
        return output
            
