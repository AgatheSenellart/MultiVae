"""To compute generative coherence of your model.

A simple usage example:

.. code-block:: python

    from multivae.metrics import CoherenceEvaluator, CoherenceEvaluatorConfig

    eval_config = CoherenceEvaluatorConfig(batch_size=128,
                                    wandb_path='your_wandb_path', #optional / to log to wandb
                                    num_classes=10, # number of classes in your multimodal dataset
                                    nb_samples_for_cross=10,
                                    nb_samples_for_joint=100
                                    )

    eval_module = CoherenceEvaluator(
        model=your_model,
        test_dataset=test_data,
        output='your_ouput_path', # where to save metrics
        sampler= None,# you can pass a trained MultiVae sampler
        classifiers=your_dict_of_classifiers
    )

    # Compute joint coherence and all cross-coherences
    eval_module.eval()

    # If you only wish to compute joint coherence:
    eval_module.joint_coherence()

    # If you want one specific cross-modal coherence
    eval_module.coherence_from_subset(['mod1', 'mod2'])

    eval_module.finish() # to finish wandb run

"""

from .coherences import CoherenceEvaluator
from .coherences_config import CoherenceEvaluatorConfig

__all__ = ["CoherenceEvaluator", "CoherenceEvaluatorConfig"]
