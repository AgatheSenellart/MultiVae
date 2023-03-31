from config1 import *



model_config = JMVAEConfig(
    n_modalities=5,
    input_dims={k: (3, 28, 28) for k in modalities},
    latent_dim=latent_dim,
    warmup=400,
)

modalities

encoders = {
    k: Encoder_ResNet_VAE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

decoders = {
    k: Decoder_ResNet_AE_MMNIST(
        BaseAEConfig(latent_dim=model_config.latent_dim, input_dim=(3, 28, 28))
    )
    for k in modalities
}

model = JMVAE(model_config, encoders=encoders, decoders=decoders)

trainer_config = BaseTrainerConfig(
    num_epochs=800,
    learning_rate=1e-4,
    steps_predict=1,
    start_keep_best_epoch=model_config.warmup + 1,
    per_device_train_batch_size=128,
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(trainer_config, model_config, project_name="mmnist")

callbacks = [TrainingCallback(), ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    training_config=trainer_config,
    callbacks=callbacks,
)
trainer.train()

# data = set_inputs_to_device(eval_data[:100], device="cuda")
# nll = model.compute_joint_nll(data)
# print(nll)
