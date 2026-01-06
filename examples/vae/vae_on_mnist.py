from multivae.data.datasets.mnist_labels import MnistLabels
from torch.utils.data import random_split
from multivae.models import CVAE, CVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback, TensorboardCallback
from pathlib import Path
from multivae.metrics import (
    Visualization,
    VisualizationConfig,
    Reconstruction,
    ReconstructionConfig,
    FIDEvaluator,
    FIDEvaluatorConfig,
)
from argparse import ArgumentParser
from multivae.models.nn.base_architectures import (
    BaseJointEncoder,
    BaseConditionalDecoder,
)
from multivae.models.nn.default_architectures import (
    Encoder_VAE_MLP,
    Decoder_AE_MLP,
    BaseAEConfig,
)

# Define architectures that ignores the covariates to have a simple VAE without conditioning.


class my_encoder(BaseJointEncoder):
    """Simple MLP architectures, ignoring the covariates"""

    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.network = Encoder_VAE_MLP(
            BaseAEConfig(input_dim=input_dim, latent_dim=latent_dim), n_hidden=2
        )

    def forward(self, inputs):
        return self.network(inputs["images"])


class my_decoder(BaseConditionalDecoder):
    """Simple MLP architectures, ignoring the covariates"""

    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.network = Decoder_AE_MLP(
            BaseAEConfig(latent_dim=latent_dim, input_dim=input_dim)
        )

    def forward(self, z, cond_mods):
        return self.network(z)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--sigma_variation", choices=["sigma_vae", "optimal_sigma_vae", "none"]
    )
    args = parser.parse_args()
    if args.sigma_variation == "none":
        args.sigma_variation = None

    # Get the data
    # Set the path where you want the data to be downloaded
    DATA_PATH = "/scratch/asenella/data"
    # Set the paths where you want the models to be saved
    SAVE_PATH = "/scratch/asenella/experiments/cvae_mnist_labels"
    trainset = MnistLabels(
        DATA_PATH, split="train", download=False, random_binarized=False
    )
    trainset, evalset = random_split(trainset, [0.8, 0.2])
    testset = MnistLabels(
        DATA_PATH, split="test", download=False, random_binarized=False
    )

    # Set the model config
    model_config = CVAEConfig(
        latent_dim=8,
        input_dims=dict(images=(1, 28, 28), labels=(10,)),
        conditioning_modalities=["labels"],
        main_modality="images",
        decoder_dist="normal",
        sigma_variation=args.sigma_variation,
    )

    model = CVAE(
        model_config,
        encoder=my_encoder(latent_dim=model_config.latent_dim, input_dim=(1, 28, 28)),
        decoder=my_decoder(latent_dim=model_config.latent_dim, input_dim=(1, 28, 28)),
    )

    # Set the trainer
    trainer_config = BaseTrainerConfig(
        output_dir=SAVE_PATH,
        learning_rate=1e-3,
        per_device_eval_batch_size=128,
        per_device_train_batch_size=128,
        num_epochs=150,
        steps_predict=1,  # log images every 1 epoch
    )

    wandb_cb = WandbCallback()
    wandb_cb.setup(
        project_name="sigma_vae_on_mnist",
        training_config=trainer_config,
        model_config=model_config,
    )

    trainer = BaseTrainer(
        model=model,
        train_dataset=trainset,
        eval_dataset=evalset,
        training_config=trainer_config,
        callbacks=[wandb_cb],
    )

    trainer.train()

    ###  Save some metrics and visualizations

    best_model = trainer._best_model
    w_path = wandb_cb.run.path
    output_dir = Path(trainer.training_dir)

    # Visualizations
    vis_config = VisualizationConfig(batch_size=64, wandb_path=w_path)
    vis_module = Visualization(
        model=best_model,
        test_dataset=testset,
        output=output_dir,
        eval_config=vis_config,
    )

    vis_module.reconstruction(modality="images")
    vis_module.unconditional_samples()
    vis_module.log_to_wandb()
    vis_module.finish()

    # Reconstruction MSE
    recon_config = ReconstructionConfig(batch_size=64, wandb_path=w_path, metric="MSE")
    recon = Reconstruction(
        model=best_model,
        eval_config=recon_config,
        test_dataset=evalset,
        output=output_dir,
    )

    recon.eval()
    recon.log_to_wandb()
    recon.finish()
