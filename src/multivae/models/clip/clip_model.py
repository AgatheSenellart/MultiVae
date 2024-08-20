from typing import Dict
import torch

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder
from torch.nn import Module, ModuleDict

from ...data.datasets import MultimodalBaseDataset
from .clip_config import CLIPConfig
from .objectives import clip_loss, return_weights




class CLIP(Module):
    def __init__(self, config: CLIPConfig, networks: Dict[str, BaseEncoder]) -> None:
        super().__init__()

        self.config = config
        self.n_modalities = config.n_modalities
        self.latent_dim = config.joint_embedding_dim
        self.set_networks(networks)
        self.weights = config.weights
        
        l1 = list(networks.keys())
        self.pairs = [(l1[i], l1[j]) for j in range(len(l1)) for i in range(j) ]
        
        self.temperatures = torch.nn.Parameter(
                torch.zeros(len(self.pairs)),
                requires_grad=True,
            )
        


    def set_networks(self, networks):
        self.networks = ModuleDict()
        assert (
            len(networks) == self.n_modalities
        ), "The number of provided networks doesn't match the number of modalities."

        for m in networks:
            if not isinstance(networks[m], BaseEncoder):
                raise AttributeError(
                    "The CLIP networks must be instances of pythae BaseEncoder class."
                )
            if networks[m].latent_dim != self.latent_dim:
                raise AttributeError(
                    "The CLIP networks must have the same latent dim as the CLIP model"
                    f" itself. ({networks[m].latent_dim} is different {self.latent_dim})"
                )
            self.networks[m] = networks[m]

    def forward(self, inputs: MultimodalBaseDataset):
        if not all([k in self.networks for k in inputs.data]):
            raise AttributeError(
                f"Some inputs keys {inputs.data.keys()} are not known modalities."
                f"The CLIP networks have modalities {self.networks.keys()}"
            )

        embeddings = dict()
        for m in inputs.data:
            embeddings[m] = (self.networks[m](inputs.data[m]).embedding)

        
        total_loss = 0
        
        for i, (m1,m2) in enumerate(self.pairs):
            
            loss = clip_loss(embeddings[m1],embeddings[m2],self.temperatures[i])
            w = return_weights(m1,m2,self.weights)
            total_loss += w * loss
            
        output = ModelOutput(loss=total_loss, metrics={})
        return output
