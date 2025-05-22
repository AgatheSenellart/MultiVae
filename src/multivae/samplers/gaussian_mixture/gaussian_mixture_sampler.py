import logging

import torch
from sklearn import mixture
from torch.utils.data import DataLoader

from multivae.data import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.models.base import ModelOutput

from ...models import BaseMultiVAE
from ..base import BaseSampler
from .gaussian_mixture_config import GaussianMixtureSamplerConfig

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class GaussianMixtureSampler(BaseSampler):
    """Fits a Gaussian Mixture in the Multimodal Autoencoder's latent space.
    If the model has several latent spaces, it fits a gaussian mixture per latent space.

    Args:
        model (BaseMultiVAE): The model to sample from.
        sampler_config (BaseSamplerConfig): An instance of BaseSamplerConfig in which any sampler's
            parameters is made available. If None a default configuration is used. Default: None.

    .. note::

        The method :class:`~multivae.samplers.GaussianMixtureSampler.fit` must be called to fit the sampler
        before sampling.

    """

    def __init__(
        self, model: BaseMultiVAE, sampler_config: GaussianMixtureSamplerConfig = None
    ):
        if sampler_config is None:
            sampler_config = GaussianMixtureSamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)
        self.n_components = sampler_config.n_components
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.name = "GaussianMixtureSampler"

    def fit(self, train_data: MultimodalBaseDataset, **kwargs):
        """Method to fit the sampler from the training data.

        Args:
            train_data (MultimodalBaseDataset): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be an instance of MultimodalBaseDataset.
        """
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=100,
            shuffle=False,
        )

        z = []
        if self.model.multiple_latent_spaces:
            mod_z = {m: [] for m in self.model.encoders}

        # Compute all embeddings
        with torch.no_grad():
            for _, inputs in enumerate(train_loader):
                inputs = set_inputs_to_device(inputs, self.device)
                output = self.model.encode(inputs)
                z.append(output.z)

                if self.model.multiple_latent_spaces:
                    for m in mod_z:
                        mod_z[m].append(output.modalities_z[m])

        z = torch.cat(z)
        if self.model.multiple_latent_spaces:
            mod_z = {m: torch.cat(mod_z[m]) for m in mod_z}

        if self.n_components > z.shape[0]:
            self.n_components = z.shape[0]
            logger.warning(
                f"Setting the number of component to {z.shape[0]} since"
                "n_components > n_samples when fitting the gmm"
            )

        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=2000,
            verbose=0,
            tol=1e-3,
        )
        gmm.fit(z.cpu().detach())
        self.gmm = gmm

        if self.model.multiple_latent_spaces:
            self.mod_gmms = dict()
            for m in mod_z:
                gmm = mixture.GaussianMixture(
                    n_components=self.n_components,
                    covariance_type="full",
                    max_iter=2000,
                    verbose=0,
                    tol=1e-3,
                )

                gmm.fit(mod_z[m].cpu().detach())
                self.mod_gmms[m] = gmm

        self.is_fitted = True

    def sample(
        self, n_samples: int = 1, batch_size: int = 500, **kwargs
    ) -> torch.Tensor:
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            save_sampler_config (bool): Whether to save the sampler config. It is saved in
                output_dir

        Returns:
            ModelOutput similar as the one returned by the encode function or generate_from_prior function.
        """
        if not self.is_fitted:
            raise ArithmeticError(
                "The sampler needs to be fitted by calling sampler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(n_samples / batch_size)
        last_batch_samples_nbr = n_samples % batch_size

        batches_sizes = [batch_size] * full_batch_nbr
        if last_batch_samples_nbr != 0:
            batches_sizes = batches_sizes + [last_batch_samples_nbr]

        z_list = []

        if self.model.multiple_latent_spaces:
            mod_z = {m: [] for m in self.model.encoders}

        for batch_size in batches_sizes:
            z = (
                torch.tensor(self.gmm.sample(batch_size)[0])
                .to(self.device)
                .type(torch.float)
            )

            z_list.append(z)

            if self.model.multiple_latent_spaces:
                for m in mod_z:
                    z = (
                        torch.tensor(self.mod_gmms[m].sample(batch_size)[0])
                        .to(self.device)
                        .type(torch.float)
                    )
                    mod_z[m].append(z)

        output = ModelOutput(z=torch.cat(z_list, dim=0))

        if self.model.multiple_latent_spaces:
            output["one_latent_space"] = False
            output["modalities_z"] = {m: torch.cat(mod_z[m]) for m in mod_z}
        else:
            output["one_latent_space"] = True

        return output
