from typing import Dict

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder
from torch.nn import Module, ModuleDict

from ...data.datasets import MultimodalBaseDataset
from .gmc_config import GMCConfig


class GMC(Module):
    def __init__(self, config: GMCConfig, processors: Dict[str, BaseEncoder], shared_encoder : BaseEncoder) -> None:
        super().__init__()

        self.config = config
        self.n_modalities = config.n_modalities
        self.latent_dim = config.embedding_dim
        self.common_dim = config.common_dim
        self.use_all_singular_values = config.use_all_singular_values
        self.set_networks(processors)
        self.set_shared_encoder(shared_encoder)


    def set_networks(self, networks):
        self.networks = ModuleDict()
        assert (
            len(networks) == self.n_modalities + 1
        ), "The number of provided processors doesn't match the number of modalities."

        if not "joint" in networks.keys():
            raise AttributeError("There must be a joint processor with the key 'joint' in the processors dictionary when defining the model.")
        
        for m in networks:
            if not isinstance(networks[m], BaseEncoder):
                raise AttributeError(
                    "The GMC processors must be instances of pythae BaseEncoder class."
                )
            if networks[m].latent_dim != self.common_dim:
                raise AttributeError(
                    f"One of the GMC processor network (modality : {m}) doesn't have the same common dim as the model"
                    f" itself. ({networks[m].latent_dim} is different {self.common_dim})"
                )
            self.networks[m] = networks[m]
            
    def set_shared_encoder(self, shared_encoder):
        if not isinstance(shared_encoder, BaseEncoder):
            raise AttributeError("The shared encoder must be an instance of pythae BaseEncoder class")
        if shared_encoder.latent_dim != self.latent_dim :
            raise AttributeError("The shared encoder must have the same latent dim as the model. "
                                 f"In the model config latent dim is {self.latent_dim} different than shared encoder latent dim {shared_encoder.latent_dim}")
        self.shared_encoder = shared_encoder
        
    def forward(self, inputs: MultimodalBaseDataset):
        """Basic training step"""
        
        # Compute all the modalities specific representations
        
        modalities_z = dict()
        
        for m in inputs:
            
            h = self.networks[m](inputs.data[m]).embedding
            z = self.shared_encoder(h).embedding
            modalities_z[m] = z
        
        # Compute the joint representation
        
        joint_z = self.networks['joint'](inputs)