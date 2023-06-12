import os
import shutil

import torch
from pythae.data.datasets import BaseDataset
from pythae.models.normalizing_flows import IAF, IAFConfig, NFModel
from pythae.trainers import BaseTrainer, BaseTrainerConfig
from torch.distributions import MultivariateNormal
from torch.nn import ModuleDict
from torch.utils.data import DataLoader

from multivae.data.utils import set_inputs_to_device
from multivae.models.base import ModelOutput

from ...models import BaseMultiVAE
from ..base.base_sampler import BaseSampler
from .iaf_sampler_config import IAFSamplerConfig


class IAFSampler(BaseSampler):
    """Fits an Inverse Autoregressive Flow in the multimodal autoencoder's latent space.
    If the model has multiple latent spaces, we fit one flow per latent space.

    Args:
        model (BaseMultiVAE): The model to sample from
        sampler_config (IAFSamplerConfig): A IAFSamplerConfig instance containing
            the main parameters of the sampler. If None, a pre-defined configuration is used.
            Default: None

    .. note::

        The method :class:`~multivae.samplers.IAFSampler.fit` must be called to fit the
        sampler before sampling.
    """

    def __init__(self, model: BaseMultiVAE, sampler_config: IAFSamplerConfig = None):
        self.is_fitted = False

        if sampler_config is None:
            sampler_config = IAFSamplerConfig()

        BaseSampler.__init__(self, model=model, sampler_config=sampler_config)

        self.flows_dims = dict(shared=model.model_config.latent_dim)
        if self.model.multiple_latent_spaces:
            self.flows_dims.update(self.model.style_dims)

        self.priors = dict()
        self.flows_models = dict()

        for key in self.flows_dims:
            self.priors[key] = MultivariateNormal(
                torch.zeros(self.flows_dims[key]).to(self.device),
                torch.eye(self.flows_dims[key]).to(self.device),
            )

            iaf_config = IAFConfig(
                input_dim=(self.flows_dims[key],),
                n_made_blocks=sampler_config.n_made_blocks,
                n_hidden_in_made=sampler_config.n_hidden_in_made,
                hidden_size=sampler_config.hidden_size,
                include_batch_norm=sampler_config.include_batch_norm,
            )

            iaf_model = IAF(model_config=iaf_config)
            self.flows_models[key] = NFModel(self.priors[key], iaf_model).to(
                self.device
            )

        self.name = "IAFsampler"

    def fit(
        self, train_data, eval_data=None, training_config: BaseTrainerConfig = None
    ):
        """Method to fit the sampler from the training data.

        Args:
            train_data (MultimodalBaseDataset): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
            eval_data (MultimodalBaseDataset): The train data needed to retreive the evaluation embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
            training_config (BaseTrainerConfig): the training config to use to fit the flow.
        """

        train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)

        zs = {m: [] for m in self.flows_models}

        try:
            with torch.no_grad():
                for _, inputs in enumerate(train_loader):
                    inputs = set_inputs_to_device(inputs, self.device)
                    encoder_output = self.model.encode(inputs)
                    zs["shared"].append(encoder_output.z)

                    if self.model.multiple_latent_spaces:
                        for m in encoder_output.modalities_z:
                            zs[m].append(encoder_output.modalities_z[m])

        except RuntimeError:
            for _, inputs in enumerate(train_loader):
                inputs = set_inputs_to_device(inputs, self.device)
                encoder_output = self.model.encode(inputs)
                zs["shared"].append(encoder_output.z.detach())

                if self.model.multiple_latent_spaces:
                    for m in encoder_output.modalities_z:
                        zs[m].append(encoder_output.modalities_z[m].detach())

        train_data = {m: torch.cat(zs[m], dim=0) for m in zs}

        # Eval dataset

        if eval_data is not None:
            eval_loader = DataLoader(dataset=eval_data, batch_size=100, shuffle=False)

            zs = {m: [] for m in self.flows_models}

            try:
                with torch.no_grad():
                    for _, inputs in enumerate(eval_loader):
                        inputs = set_inputs_to_device(inputs, self.device)
                        encoder_output = self.model.encode(inputs)
                        zs["shared"].append(encoder_output.z)
                        if self.model.multiple_latent_spaces:
                            for m in encoder_output.modalities_z:
                                zs[m].append(encoder_output.modalities_z[m])

            except RuntimeError:
                for _, inputs in enumerate(train_loader):
                    inputs = set_inputs_to_device(inputs, self.device)
                    encoder_output = self.model.encode(inputs)
                    zs["shared"].append(encoder_output.z.detach())
                    if self.model.multiple_latent_spaces:
                        for m in encoder_output.modalities_z:
                            zs[m].append(encoder_output.modalities_z[m].detach())

            eval_data = {m: torch.cat(zs[m]) for m in zs}

        self.iaf_models = torch.nn.ModuleDict()

        for m in train_data:  # number of latent_spaces
            train_dataset = BaseDataset(
                data=train_data[m], labels=torch.zeros((len(train_data[m]),))
            )
            eval_dataset = (
                None
                if eval_data is None
                else BaseDataset(
                    data=eval_data[m], labels=torch.zeros((len(eval_data[m]),))
                )
            )

            trainer = BaseTrainer(
                model=self.flows_models[m],
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=training_config,
            )

            trainer.train()

            self.iaf_models[m] = IAF.load_from_folder(
                os.path.join(trainer.training_dir, "final_model")
            ).to(self.device)

            shutil.rmtree(trainer.training_dir)

        self.is_fitted = True

    def sample(
        self, n_samples: int = 1, batch_size: int = 500, **kwargs
    ) -> torch.Tensor:
        """Main sampling function of the sampler.

        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated
                data. Default: True.
            save_sampler_config (bool): Whether to save the sampler config. It is saved in
                output_dir

        Returns:
            ~torch.Tensor: The generated images
        """

        if not self.is_fitted:
            raise ArithmeticError(
                "The sampler needs to be fitted by calling sampler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(n_samples / batch_size)
        last_batch_samples_nbr = n_samples % batch_size
        batches = [batch_size] * full_batch_nbr
        if last_batch_samples_nbr != 0:
            batches = batches + [last_batch_samples_nbr]

        z_gen = {m: [] for m in self.iaf_models}

        for batch in batches:
            for m in self.iaf_models:
                u = self.priors[m].sample((batch,))
                z = self.iaf_models[m].inverse(u).out
                z_gen[m].append(z)

        # Output with the same format as the output of encode or generate_from_prior functions
        output = ModelOutput(
            z=torch.cat(z_gen.pop("shared")),
            one_latent_space=not self.model.multiple_latent_spaces,
        )
        if self.model.multiple_latent_spaces:
            output["modalities_z"] = {m: torch.cat(z_gen[m]) for m in z_gen}

        return output
