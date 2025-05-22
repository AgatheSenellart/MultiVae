"""This module performs latent clustering with k-means in the latent space
and returns the clustering accuracy.

Basic code example:

.. code-block:: python

    from multivae.metrics import Clustering, ClusteringConfig

    eval_config = ClusteringConfig(batch_size=128,
                                wandb_path='your_wandb_path',
                                n_clusters=10,
                                number_of_runs=10)

    eval_module = Clustering(
        model = your_model,
        test_dataset=test_set,
        train_dataset=train_data,
        output='./metrics',# where to save metrics
        eval_config=eval_config
    )

    # Compute clustering accuracy
    eval_module.eval()

    eval_module.finish() # finishes wandb run


"""

from .clustering_class import Clustering
from .clustering_config import ClusteringConfig
