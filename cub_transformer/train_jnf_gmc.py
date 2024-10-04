
from multivae.models import JNFGMC, JNFGMCConfig, GMC, GMCConfig
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--param_file", type=str)
args = parser.parse_args()

with open(args.param_file, "r") as fp:
    info = json.load(fp)
args = argparse.Namespace(**info)


# GMC model
gmc_config = GMCConfig(
    n_modalities=2,
    common_dim=64,
    latent_dim=64,
    temperature=0.1,
    loss= args.loss)


gmc_model = GMC(gmc_config,
                processors=dict(
                    image = EncoderImg(0,gmc_config.common_dim,'normal'),
                    text = CubTextEncoder(
                        latent_dim=gmc_config.common_dim,
                        max_sentence_length=max_length,
                        ntokens=vocab_size,
                        embed_size=512
                        )),
                shared_encoder= CUBCommonEncoder(gmc_config.common_dim, latent_dim=gmc_config.latent_dim))
                

# model
model_config = JNFGMCConfig(
    n_modalities=2,
    latent_dim=64,
    uses_likelihood_rescaling=True,

    rescale_factors=dict(image = max_length/(3*64*64),
                         text = 5.0),
    
    decoders_dist=dict(image = 'laplace',
                       text ='categorical'),
    
    decoder_dist_params=dict(image = dict(scale=0.01)),
    nb_epochs_gmc=150,
    warmup=args.warmup,
    annealing=args.annealing,
    alpha=args.alpha,
    beta=args.beta
    
)

encoders = dict(
    image = Encoder_VAE_MLP(BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=model_config.latent_dim)),
    text = Encoder_VAE_MLP(BaseAEConfig(input_dim=(gmc_config.latent_dim,), latent_dim=model_config.latent_dim))
)

joint_encoder = MultipleHeadJointEncoder(
    dict(image = EncoderImg(0,model_config.latent_dim,'normal'),
         text = CubTextEncoder(
             latent_dim=model_config.latent_dim,
             max_sentence_length=max_length,
             ntokens=vocab_size,
             embed_size=512,
             dropout=0.2
         ))
    , BaseAEConfig(latent_dim=model_config.latent_dim)
)

decoders = dict(
    image = DecoderImg(model_config.latent_dim),
    text = CubTextDecoderMLP(
        BaseAEConfig(input_dim=(max_length,vocab_size),latent_dim=model_config.latent_dim)
    )
)


model=JNFGMC(model_config=model_config,
                encoders = encoders, 
                joint_encoder=joint_encoder,
                decoders=decoders,
                gmc_model=gmc_model
                )



# trainer and callbacks
training_config = MultistageTrainerConfig(
    output_dir=save_path,
    per_device_eval_batch_size=64,
    per_device_train_batch_size=64,
    num_epochs= model_config.nb_epochs_gmc + model_config.warmup + 150,
    optimizer_cls="Adam",
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 20},
    learning_rate=1e-3,
    steps_predict=10,
    seed=args.seed
    
)

wandb = WandbCallback()
wandb.setup(training_config=training_config,model_config=model_config, project_name="CUB_transformer")

trainer = MultistageTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    callbacks=[wandb],
    training_config=training_config
    
)

trainer.train()

# Validate and compute coherence
from evaluate_coherence import evaluate_coherence
test_data = CUB(data_path, split='test',max_lenght=32).text_data
model = trainer._best_model
wandb_path = wandb.run._get_path()
evaluate_coherence(model, wandb_path,test_data)
