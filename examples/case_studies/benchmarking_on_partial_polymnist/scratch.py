from multivae.metrics import Reconstruction, ReconstructionConfig

eval_config = ReconstructionConfig(
                               batch_size=128,
                               wandb_path='your_wandb_path',
                               metric='SSIM' # take ten datapoints for conditional generation
                               )

eval_module = Reconstruction(
    model = your_model,
    test_dataset=test_set,
    output='./metrics',# where to save images
    eval_config=eval_config,
)

# Compute metrics
eval_module.eval()

eval_module.finish() # finishes wandb run