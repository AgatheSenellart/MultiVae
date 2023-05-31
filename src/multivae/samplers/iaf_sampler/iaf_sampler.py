import os
import shutil

import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from ...models import BaseMultiVAE
from pythae.models.normalizing_flows import IAF, IAFConfig, NFModel
from ...trainers import BaseTrainer, BaseTrainerConfig
from ..base.base_sampler import BaseSampler
from .iaf_sampler_config import IAFSamplerConfig
from torch.nn import ModuleDict
from torch.utils.data import TensorDataset

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

        self.prior = MultivariateNormal(
            torch.zeros(model.model_config.latent_dim).to(self.device),
            torch.eye(model.model_config.latent_dim).to(self.device),
        )

        iaf_config = IAFConfig(
            input_dim=(model.model_config.latent_dim,),
            n_made_blocks=sampler_config.n_made_blocks,
            n_hidden_in_made=sampler_config.n_hidden_in_made,
            hidden_size=sampler_config.hidden_size,
            include_batch_norm=sampler_config.include_batch_norm,
        )

        iaf_model = IAF(model_config=iaf_config)
        self.flow_contained_model = NFModel(self.prior, iaf_model)
        self.flow_contained_model.to(self.device)
        
        if model.multiple_latent_spaces:
            self.additional_flows = dict()
            for m in model.encoders:
                iaf_model = IAF(model_config=iaf_config)
                self.additional_flows[m] = NFModel(self.prior, iaf_model)
                self.additional_flows[m].to(self.device)

    def fit(
        self, train_data, eval_data=None, training_config: BaseTrainerConfig = None
    ):
        """Method to fit the sampler from the training data

        Args:
            train_data (MultimodalBaseDataset): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
            eval_data (MultimodalBaseDataset): The train data needed to retreive the evaluation embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x
                    ... and in range [0-1]
            training_config (BaseTrainerConfig): the training config to use to fit the flow.
        """

        assert (
            train_data.max() >= 1 and train_data.min() >= 0
        ), "Train data must in the range [0-1]"


        train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)

        z = []
        mod_z = {m:[] for m in self.model.encoders}
        
        try:
            with torch.no_grad():
                for _, inputs in enumerate(train_loader):
                    encoder_output = self.model.encode(inputs)
                    z_ = encoder_output.z
                    z.append(z_)
                    if self.model.multiple_latent_spaces:
                        for m in mod_z:
                            mod_z[m].append(encoder_output.modalities_z[m])

        except RuntimeError:
            for _, inputs in enumerate(train_loader):
                    encoder_output = self.model.encode(inputs)
                    z_ = encoder_output.z.detach()
                    z.append(z_)
                    if self.model.multiple_latent_spaces:
                        for m in mod_z:
                            mod_z[m].append(encoder_output.modalities_z[m].detach())


        train_data = dict(shared= torch.cat(z))
        if self.model.multiple_latent_spaces:
            train_data.update({m : torch.cat(mod_z[m]) for m in mod_z})

        # Eval dataset 

        if eval_data is not None:

            assert (
                eval_data.max() >= 1 and eval_data.min() >= 0
            ), "Eval data must in the range [0-1]"


            eval_loader = DataLoader(
                dataset=eval_data, batch_size=100, shuffle=False
            )

            z = []
            mod_z = {m:[] for m in self.model.encoders}
            
            try:
                with torch.no_grad():
                    for _, inputs in enumerate(eval_loader):
                        encoder_output = self.model.encode(inputs)
                        z_ = encoder_output.z
                        z.append(z_)
                        if self.model.multiple_latent_spaces:
                            for m in mod_z:
                                mod_z[m].append(encoder_output.modalities_z[m])

            except RuntimeError:
                for _, inputs in enumerate(train_loader):
                        encoder_output = self.model.encode(inputs)
                        z_ = encoder_output.z.detach()
                        z.append(z_)
                        if self.model.multiple_latent_spaces:
                            for m in mod_z:
                                mod_z[m].append(encoder_output.modalities_z[m].detach())


            eval_data = dict(shared= torch.cat(z))
            if self.model.multiple_latent_spaces:
                eval_data.update({m : torch.cat(mod_z[m]) for m in mod_z})

        self.iaf_models = torch.nn.ModuleDict()

        for m in train_data: # number of latent_spaces
            
            train_dataset = TensorDataset(train_data[m])
            eval_dataset = None if eval_data is None else TensorDataset(eval_data[m])
            
            trainer = BaseTrainer(
                model=self.flow_contained_model,
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
        self,
        num_samples: int = 1,
        batch_size: int = 500,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
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

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            u = self.prior.sample((batch_size,))
            z = self.iaf_model.inverse(u).out
            x_gen = self.model.decoder(z).reconstruction.detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            u = self.prior.sample((last_batch_samples_nbr,))
            z = self.iaf_model.inverse(u).out
            x_gen = self.model.decoder(z).reconstruction.detach()

            if output_dir is not None:
                for j in range(last_batch_samples_nbr):
                    self.save_img(
                        x_gen[j],
                        output_dir,
                        "%08d.png" % int(batch_size * full_batch_nbr + j),
                    )

            x_gen_list.append(x_gen)

        if save_sampler_config:
            self.save(output_dir)

        if return_gen:
            return torch.cat(x_gen_list, dim=0)
