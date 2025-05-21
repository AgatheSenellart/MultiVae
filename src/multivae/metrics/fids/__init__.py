"""This module allow to easily compute FID metrics on a MultiVae model.

We are grateful to https://github.com/mseitzer/pytorch-fid, on which our code is heavily based.

A simple example:

.. code-block:: python

    from multivae.metrics import FIDEvaluator, FIDEvaluatorConfig

    fid_config = FIDEvaluatorConfig(batch_size=128,
                                    inception_weights_path='your_path',
                                    wandb_path='your_wandb_path' #optional / to log to wandb
                                    )

    fid_module = FIDEvaluator(
        model=your_model,
        test_dataset=test_data,
        output='your_ouput_path', # where to save metrics
        sampler= None,# you can pass a trained MultiVae sampler
        custom_encoders=None,# If you wish to use custom networks for each modality rather than the inception network
    )

    # Compute FID for unconditional generation
    fid_module.eval()

    # Compute FID for conditional generation
    fid_module.compute_all_conditional_fids(gen_mod = 'modality_to_generate')


"""

from .fids import FIDEvaluator
from .fids_config import FIDEvaluatorConfig

__all__ = ["FIDEvaluator", "FIDEvaluatorConfig"]
