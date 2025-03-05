"""This module computes likelihood for any multivae model.

.. code-block:: python

    from multivae.metrics import LikelihoodsEvaluator, LikelihoodsEvaluatorConfig

    eval_config = LikelihoodsEvaluatorConfig(batch_size=128,
                                wandb_path='your_wandb_path',
                                num_samples=1000)

    eval_module = LikelihoodsEvaluator(
        model = your_model,
        test_dataset=test_set,
        output='./metrics',# where to save metrics
        eval_config=eval_config
    )

    # Compute joint negative log likelihood
    eval_module.eval()

    eval_module.finish() # finishes wandb run


"""

from .likelihoods import LikelihoodsEvaluator
from .likelihoods_config import LikelihoodsEvaluatorConfig

__all__ = ["LikelihoodsEvaluator", "LikelihoodsEvaluatorConfig"]
