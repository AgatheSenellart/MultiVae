"""Base sampler class, adapted from Pythae Base Sampler."""

import logging
import os

import torch

from ...data.datasets.base import MultimodalBaseDataset
from ...models import BaseMultiVAE
from .base_sampler_config import BaseSamplerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseSampler:
    """Base class for samplers used to generate from the MultiVae models' joint latent spaces.

    Args:
        model (BaseMultivae): The model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None
    """

    def __init__(self, model: BaseMultiVAE, sampler_config: BaseSamplerConfig = None):
        if sampler_config is None:
            sampler_config = BaseSamplerConfig()

        self.model = model
        self.model.eval()
        self.sampler_config = sampler_config
        self.is_fitted = False

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model.device = device

        self.model.to(device)
        self.name = "BaseSampler"

    def fit(self, train_data: MultimodalBaseDataset, **kwargs):
        """Function to be called to fit the sampler before sampling."""
        return

    def sample(
        self,
        n_samples: int = 1,
        batch_size: int = 500,
        return_gen: bool = True,
    ):
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            return_gen (bool): Whether the sampler should directly return a the generated
                data. Default: True.

        Returns:
            ~torch.Tensor: The generated images
        """
        raise NotImplementedError()

    def save(self, dir_path):
        """Method to save the sampler config. The config is saved a as ``sampler_config.json``
        file in ``dir_path``.
        """
        logger.info("Saving model in %s.", dir_path)

        os.makedirs(dir_path, exist_ok=True)
        self.sampler_config.save_json(dir_path, "sampler_config")
