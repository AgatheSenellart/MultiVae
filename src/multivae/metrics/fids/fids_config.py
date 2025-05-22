from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class FIDEvaluatorConfig(EvaluatorConfig):
    """Config class for the evaluation of the coherences module.

    Args:
        batch_size (int) : The batch size to use in the evaluation. Default to 512
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            See :doc:`info_wandb` for more information.
            If None is provided, the metrics are not logged on wandb.
            Default to None.
        inception_weights_path (str) : The path to InceptionV3 weights. Default to
            '../fid_model/model.pt'.
        dims_inception (int) : Select the embedding layer of the Inception network
            defined by its output_size. Default to 2048.

    """

    inception_weights_path: str = "../fid_model/model.pt"
    dims_inception: int = 2048
