"""This module can be use to generate, visualize and save samples with any MultiVae model.

Basic usage example:

.. code-block::

    from multivae.metrics import Visualization, VisualizationConfig

    eval_config = VisualizationConfig(
                                wandb_path='your_wandb_path',
                                n_data_cond=10, # take ten datapoints for conditional generation
                                n_samples=5, # generate 5 samples per datapoint
                                )

    eval_module = Visualization(
        model = your_model,
        test_dataset=test_set,
        output='./metrics',# where to save images
        eval_config=eval_config,
        sampler=None # you can use a trained MultiVae sampler for joint generation
    )

    # Generate unconditional samples
    eval_module.eval()

    # Generate conditional samples from a subset of modalities
    eval_module.conditional_samples_subset(subset=['modality_1', 'modality_2'], gen_mod='all')

    eval_module.finish() # finishes wandb run

"""

from .visualization_class import Visualization
from .visualize_config import VisualizationConfig

__all__ = ["Visualization", "VisualizationConfig"]
