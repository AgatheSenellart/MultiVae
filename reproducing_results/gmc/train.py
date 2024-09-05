from multivae.models.gmc import GMC, GMCConfig
from multivae.data.datasets import MHD
from architectures import *
from torch.optim import Adam
from tqdm import tqdm
import wandb


dataset = MHD('/Users/agathe/dev/data/MHD')

model_config = GMCConfig(
    n_modalities=4,
    input_dims = dict(image = (1,28,28), audio = (1,32,28), trajectory = (200,), label=(10,)),
    common_dim=64,
    latent_dim=64,
    temperature=0.1
)

model = GMC(
    config=model_config,
    processors = dict(image = MHDImageProcessor(model_config.common_dim),
                      audio = MHDSoundProcessor(model_config.common_dim),
                      trajectory= MHDTrajectoryProcessor(common_dim=model_config.common_dim),
                      label = MHDLabelProcessor(model_config.common_dim)
                      ),
    joint_encoder=MHDJointProcessor(model_config.common_dim),
    shared_encoder=MHDCommonEncoder(model_config.common_dim,model_config.latent_dim)
)


from torch.utils.data import DataLoader

dl = DataLoader(dataset,64)

optimizer = Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    epoch_loss = 0
    for i, batch in enumerate(tqdm(dl)) :
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    print("loss : ", epoch_loss)

# Save weights

save_dir = '/home/asenella/scratch/'

for m in model.networks:
    torch.save(model.networks[m].state_dict(),save_dir + 'networks')





