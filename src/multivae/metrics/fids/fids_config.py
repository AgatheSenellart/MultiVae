from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class FIDEvaluatorConfig(EvaluatorConfig):
    """

    Config class for the evaluation of the coherences module.

    Args:
        batch_size (int) : The batch size to use in the evaluation. Default to 512
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            For an existing run (the training run), the info can be found in the training dir (in wandb_info.json)
            at the end of training (if wandb was used) or on the hugging_face webpage of the run.
            Otherwise the user can create a new wandb run and get the path with :
                `
                import wandb
                run = wandb.init(entity = your_entity, project=your_project)
                wandb_path = run.path

            If None are provided, the metrics are not logged on wandb.
            Default to None.
        inception_weights_path (str) : The path to InceptionV3 weights. Default to
            '../fid_model/model.pt'.
        dims_inception (int) : Select the embedding layer of the Inception network
            defined by its output_size. Default to 2048.

    """

    inception_weights_path: str = "../fid_model/model.pt"
    dims_inception: int = 2048
