from multivae.data import MultimodalBaseDataset
from multivae.metrics.base.evaluator_config import EvaluatorConfig
from multivae.models.base import BaseMultiVAE
from multivae.samplers.base import BaseSampler
import torch
from .visualize_config import VisualizationConfig
from ..base.evaluator_class import Evaluator
from multivae.data.datasets.utils import adapt_shape
from torchvision.utils import make_grid
from PIL import Image
import os
import wandb
from multivae.models.base import ModelOutput

class Visualization(Evaluator):
    
    """_summary_
    """
    
    def __init__(self, model: BaseMultiVAE, test_dataset: MultimodalBaseDataset, output: str = None, eval_config=VisualizationConfig(), sampler: BaseSampler = None) -> None:
        super().__init__(model, test_dataset, output, eval_config, sampler)
        
        
        
    
    def unconditional_samples(self,n_samples):
        
        
        if self.sampler is None:
            samples = self.model.generate_from_prior(n_samples)
        else:
            samples = self.sampler.sample(n_samples)
        
        images = self.model.decode(samples)
        recon, shape = adapt_shape(images)

        recon_image = torch.cat(list(recon.values()))

        # Transform to PIL format
        recon_image = make_grid(recon_image, nrow=n_samples)
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
            recon_image.save(os.path.join(self.output, 'unconditional.png'))

        if self.wandb_run is not None:
            self.wandb_run.log({'unconditional_generation' : wandb.Image(recon_image)})
            
        return recon_image
    
    
    def eval(self, n_samples = 5):
        
        image  = self.unconditional_samples(n_samples)
        
        return ModelOutput(unconditional_generation = image)
    