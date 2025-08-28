from multivae.data.datasets.mnist_labels import MnistLabels
from torch.utils.data import random_split
from multivae.models import CVAE, CVAEConfig
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import WandbCallback, TensorboardCallback
from pathlib import Path
from multivae.metrics import Visualization, VisualizationConfig
from argparse import ArgumentParser

if __name__=='__main__':
    
    parser=ArgumentParser()
    parser.add_argument('--sigma_variation', choices=['sigma_vae', 'optimal_sigma_vae', 'none'])
    args = parser.parse_args()
    if args.sigma_variation=='none':
        args.sigma_variation=None

    # Get the data
    # Set the path where you want the data to be downloaded 
    DATA_PATH = '/home/asenella/data'
    # Set the paths where you want the models to be saved
    SAVE_PATH = '/home/asenella/experiments/cvae_mnist_labels'

    trainset = MnistLabels(DATA_PATH,split='train',download=False )
    trainset, evalset = random_split(trainset, [0.8, 0.2])
    testset = MnistLabels(DATA_PATH,split='test',download=False )


    # Set the model config
    model_config = CVAEConfig(
        latent_dim=16,
        input_dims=dict(images=(1, 28, 28), labels=(10,)),
        # conditioning_modalities=['labels'],
        conditioning_modalities=[],
        main_modality='images',
        decoder_dist='normal',
        sigma_variation=args.sigma_variation
    )
        
    model = CVAE(model_config)

    # Set the trainer
    trainer_config = BaseTrainerConfig(
        output_dir=SAVE_PATH,
        learning_rate=1e-3,
        per_device_eval_batch_size=128,
        per_device_train_batch_size=128,
        num_epochs=30,
        steps_predict=1 # log images every 1 epoch
        )

    wandb_cb = WandbCallback()
    wandb_cb.setup(project_name='sigma_vae_on_mnist', training_config=trainer_config, model_config=model_config)


    trainer=BaseTrainer(
        model=model,
        train_dataset=trainset,
        eval_dataset=evalset,
        training_config=trainer_config, 
        callbacks=[wandb_cb]
        
    )

    trainer.train()

    # Save some metrics and visualizations
    vis_module = Visualization(
        model=trainer._best_model,
        test_dataset=testset,
        output=Path(trainer.training_dir) / 'visualisations'
        )

    vis_module.reconstruction(modality='images')
    vis_module.unconditional_samples()