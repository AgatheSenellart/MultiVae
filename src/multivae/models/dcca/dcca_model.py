from typing import Dict

from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseEncoder
from torch.nn import Module, ModuleDict

from ...data.datasets import MultimodalBaseDataset
from .dcca_config import DCCAConfig
from .objectives import cca_loss, mcca_loss


class DCCA(Module):
    def __init__(self, config: DCCAConfig, networks: Dict[str, BaseEncoder]) -> None:
        super().__init__()

        self.config = config
        self.n_modalities = config.n_modalities
        self.latent_dim = config.embedding_dim
        self.use_all_singular_values = config.use_all_singular_values
        self.set_networks(networks)

        if self.config.n_modalities == 2:
            self.loss = cca_loss(self.latent_dim, self.use_all_singular_values).loss
        else:
            self.loss = mcca_loss(self.latent_dim, self.use_all_singular_values).loss

    def set_networks(self, networks):
        self.networks = ModuleDict()
        assert (
            len(networks) == self.n_modalities
        ), "The number of provided networks doesn't match the number of modalities."

        for m in networks:
            if not isinstance(networks[m], BaseEncoder):
                raise AttributeError(
                    "The DCCA networks must be instances of pythae BaseEncoder class."
                )
            if networks[m].latent_dim != self.latent_dim:
                raise AttributeError(
                    "The DCCA networks must have the same latent dim as the DCCA model"
                    f" itself. ({networks[m].latent_dim} is different {self.latent_dim})"
                )
            self.networks[m] = networks[m]

    def forward(self, inputs: MultimodalBaseDataset):
        if not all([k in self.networks for k in inputs.data]):
            raise AttributeError(
                f"Some inputs keys {inputs.data.keys()} are not known modalities."
                f"The DCCA networks have modalities {self.networks.keys()}"
            )

        embeddings = []
        for m in inputs.data:
            embeddings.append(self.networks[m](inputs.data[m]).embedding)

        # Compute CCA loss or MultiCCA loss between the embeddings

        loss = self.loss(embeddings)

        output = ModelOutput(loss=loss, metrics={})
        return output
