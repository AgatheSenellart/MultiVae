import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from multivae.data import MultimodalBaseDataset
from multivae.data.datasets.utils import adapt_shape
from multivae.data.utils import set_inputs_to_device
from multivae.metrics.base.evaluator_config import EvaluatorConfig
from multivae.models.base import BaseMultiVAE, ModelOutput
from multivae.samplers.base import BaseSampler

from ..base.evaluator_class import Evaluator
from .visualize_config import VisualizationConfig


class Visualization(Evaluator):
    """
    Visualization Module for visualizing unconditional, conditional samples from models.

    Args:

        model (BaseMultiVAE) : the model to evaluate.
        test_dataset (MultimodalBaseDataset) : the dataset to use for conditional image generation.
        output (str): the path where to save images and metrics. Default to None.
        eval_config (VisualizationConfig) : The configuration file for this evaluation module.
            Optional.
        sampler (BaseSampler) : The sampler to use for joint generation. Optional. If None is
            provided, the sampler is used.
            
    .. code-block::

            >>> from multivae.metrics.visualization import Visualization, VisualizationConfig


            >>> vis_config = VisualizationConfig(
            ...                    wandb_path='your_wandb_path', # optional, if you have initialized a wandb run
            ...                     n_samples=5, # number of generated samples
            ...                     n_data_cond=8, # For conditional generation, the number of datapoints to use.
            ...                     )

            >>> vis_module = Visualization(
            ...                    model, 
            ...                    test_dataset=test_set,
            ...                    output='./metrics',
            ...                    eval_config=vis_config)
            
            # Compute conditional generations
            >>> generations = vis_module.conditional_samples_subset(['name_of_conditioning_modality1'])

            # Compute unconditional generations
            >>> generations = vis_module.unconditional_samples()
            
            
    
    
    """

    def __init__(
        self,
        model: BaseMultiVAE,
        test_dataset: MultimodalBaseDataset,
        output: str = None,
        eval_config=VisualizationConfig(),
        sampler: BaseSampler = None,
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config, sampler)
        self.n_samples = eval_config.n_samples
        self.n_data_cond = eval_config.n_data_cond

    def unconditional_samples(self, **kwargs):
        device = kwargs.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.sampler is None:
            samples = self.model.generate_from_prior(self.n_samples)
        else:
            samples = self.sampler.sample(self.n_samples)
        from multivae.data.utils import set_inputs_to_device

        samples = set_inputs_to_device(samples, device=device)
        recon = self.model.decode(samples)
        
        if hasattr(self.test_dataset, "transform_for_plotting"):
            recon = {m : self.test_dataset.transform_for_plotting(recon[m], m) for m in recon}
        
        recon, shape = adapt_shape(recon)

        recon_image = torch.cat(list(recon.values()))

        # Transform to PIL format
        recon_image = make_grid(recon_image, nrow=self.n_samples)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            recon_image.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        recon_image = Image.fromarray(ndarr)

        if self.output is not None:
            recon_image.save(os.path.join(self.output, "unconditional.png"))

        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log({"unconditional_generation": wandb.Image(recon_image)})

        return recon_image

    def conditional_samples_subset(self, subset, gen_mod="all", shuffle=True,seed=0):
        torch.manual_seed(seed=seed)
        dataloader = DataLoader(self.test_dataset, batch_size=self.n_data_cond, shuffle=shuffle,)
        data = next(iter(dataloader))
        # set inputs to device
        # data = set_inputs_to_device(data, self.device)

        recon = self.model.predict(
            data,
            subset,
            gen_mod,
            N=self.n_samples,
            flatten=True,
            ignore_incomplete=True,
        )
        
        if hasattr(self.test_dataset, "transform_for_plotting"):
            recon = {m : self.test_dataset.transform_for_plotting(recon[m], m) for m in recon}
            recon.update({f"original_{m}": self.test_dataset.transform_for_plotting(data.data[m],m) for m in subset})
        else:
            recon.update({f"original_{m}": data.data[m] for m in subset})

        recon, shape = adapt_shape(recon)
        recon_image = [recon[f"original_{m}"] for m in subset]

        recon_image = recon_image + [recon[m] for m in recon if "original" not in m]
        recon_image = torch.cat(recon_image)

        # Transform to PIL format
        recon_image = make_grid(recon_image, nrow=self.n_data_cond)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = (
            recon_image.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        recon_image = Image.fromarray(ndarr)

        if self.output is not None:
            recon_image.save(
                os.path.join(self.output, f"conditional_from_subset_{subset}_to_{gen_mod}.png")
            )

        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log(
                {f"conditional_from_subset_{subset}_to_{gen_mod}": wandb.Image(recon_image)}
            )

        return recon_image

    def eval(self):
        image = self.unconditional_samples()

        return ModelOutput(unconditional_generation=image)
