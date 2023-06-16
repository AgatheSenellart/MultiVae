from pydantic.dataclasses import dataclass

from ..base.evaluator_config import EvaluatorConfig


@dataclass
class VisualizationConfig(EvaluatorConfig):

    """

    Config class for the visualization module.

    Args:
        batch_size (int) : The batch size to use in the evaluation. Default to 512
        wandb_path (str) : The user can provide the path of the wandb run with a
            format 'entity/projet_name/run_id' where the metrics should be logged.
            For an existing run (the training run), the info can be found in the training dir
            (in wandb_info.json)
            at the end of training (if wandb was used) or on the hugging_face webpage of the run.
            Otherwise the user can create a new wandb run and get the path with :
                `
                import wandb
                run = wandb.init(entity = your_entity, project=your_project)
                wandb_path = run.path

            If None are provided, the metrics are not logged on wandb.
            Default to None.
        n_samples(int) : The number of samples to generate per modality and per data_point for
            conditional generation. Default to 5.
        n_data_cond (int) : The number of datapoints to use for conditional generation. Default to 5
    """

    n_samples: int = 5
    n_data_cond: int = 5
