from multivae.models import Nexus, NexusConfig
from multivae.data.datasets.mhd import MHD
from multivae.models.base import BaseEncoder, BaseAEConfig, BaseDecoder, ModelOutput
from torch import nn 
import torch
from math import prod
from torch.utils.data import DataLoader
import torch.nn.functional as F
from multivae.trainers import BaseTrainer, BaseTrainerConfig
from multivae.trainers.base.callbacks import (
    ProgressBarCallback,
    WandbCallback,
)

# import the datasets
train_set = MHD('/home/asenella/scratch/data/MHD', split='train', modalities=['audio', 'trajectory', 'image', 'label'])
test_set = MHD('/home/asenella/scratch/data/MHD', split='test', modalities=['audio', 'trajectory', 'image', 'label'])

# define the architectures

def convolutional_output_width(input_width, kernel_width, padding, stride):
    # assumes square input/output and kernels
    return int((input_width - kernel_width + 2 * padding) / stride + 1)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Bottom encoders
class ImageEncoder(BaseEncoder):
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.conv_layers = nn.Sequential( 
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish()
        )
        
       
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            Swish(),
            nn.Linear(128,128),
            Swish()
        )
   
        # Output layer of the network
        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_logvar = nn.Linear(128, self.latent_dim) 
        
    def forward(self, x):
        x = self.conv_layers(x)
        h = x.view(x.size(0), -1)
        h = self.linear_layers(h)
        return ModelOutput(embedding = self.fc_mu(h), log_covariance = self.fc_logvar(h))  
    
class SoundEncoder(BaseEncoder):
    def __init__(self, output_dim):
        super(SoundEncoder, self).__init__()
        self.latent_dim = output_dim

        # Properties
        self.conv_layer_0 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.fc_mu = nn.Linear(2048, output_dim)
        self.fc_logvar = nn.Linear(2048, output_dim)


    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        h = x.view(x.size(0), -1)
        return ModelOutput(
            embedding = self.fc_mu(h),
            log_covariance = self.fc_logvar(h))

class TrajectoryEncoder(BaseEncoder):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(TrajectoryEncoder, self).__init__()
        self.latent_dim = output_dim

        # Variables
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        enc_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]
            enc_layers.append(nn.Linear(pre, pos))
            enc_layers.append(nn.BatchNorm1d(pos))
            enc_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        # Output layer of the network
        self.fc_mu = nn.Linear(pre, output_dim)
        self.fc_logvar = nn.Linear(pre, output_dim)

        # Print information
        print(f'Layers: {enc_layers}')
        self.network = nn.Sequential(*enc_layers)

    def forward(self, x):
        h = self.network(x)
        return ModelOutput(embedding = self.fc_mu(h),
                           log_covariance= self.fc_logvar(h))

class LabelEncoder(BaseEncoder):
    
    def __init__(self, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(10,128), nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.LeakyReLU(),
            nn.Linear(128,128), nn.BatchNorm1d(128), nn.LeakyReLU(),
        )
        
        self.fc_embedding = nn.Linear(128,latent_dim)
        self.fc_covariance = nn.Linear(128,latent_dim)
    
    def forward(self, x):
        h = self.layers(x)
        return ModelOutput(
            embedding = self.fc_embedding(h),
            log_covariance = self.fc_covariance(h)
        )
    
    
# Bottom decoders

class ImageDecoder(BaseDecoder):
    def __init__(self,input_dim, latent_dim):
        super(ImageDecoder, self).__init__()

        # Variables
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Linear layers =
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.latent_dim, 128), Swish(),
            nn.Linear(128, 128), Swish(),
            nn.Linear(128,3136), Swish(),
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(
                    64, 32, 4, 2, 1, bias=False), Swish(),
            nn.ConvTranspose2d(
                    32, 1, 4, 2, 1, bias=False), nn.Sigmoid(),
        )
        
    def forward(self, z):
        x = self.linear_layers(z)
        x = x.view(-1, 64, 7,7)
        out = self.conv_layers(x)
        return ModelOutput(reconstruction = out)

class SoundDecoder(BaseDecoder):
    def __init__(self, input_dim):
        super(SoundDecoder, self).__init__()
        self.latent_dim = input_dim

        self.upsampler = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.hallucinate_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
        )


    def forward(self, z):
        
        batch_shape = z.shape[:-1]
        z = z.reshape(prod(batch_shape),-1)
        
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        z = self.hallucinate_0(z)
        z = self.hallucinate_1(z)
        out = self.hallucinate_2(z)
        
        if len(batch_shape) >1:
            out = out.reshape(*batch_shape, *out.shape[1:])
        
        return ModelOutput(reconstruction = F.sigmoid(out))
        
class TrajectoryDecoder(BaseDecoder):
    def __init__(self, input_dim, layer_sizes, output_dim):
        super(TrajectoryDecoder, self).__init__()
        self.latent_dim = input_dim

        # Variables
        self.id = id
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim

        # Create Network
        dec_layers = []
        pre = input_dim

        for i in range(len(layer_sizes)):
            pos = layer_sizes[i]

            # Check for input transformation
            dec_layers.append(nn.Linear(pre, pos))
            dec_layers.append(nn.BatchNorm1d(pos))
            dec_layers.append(nn.LeakyReLU())

            # Check for input transformation
            pre = pos

        dec_layers.append(nn.Linear(pre, output_dim))
        self.network = nn.Sequential(*dec_layers)

        # Output Transformation
        self.out_process = nn.Sigmoid()

        # Print information
        print(f'Layers: {dec_layers}')


    def forward(self, x):
        
        batch_shape = x.shape[:-1]
        
        x = x.reshape(prod(batch_shape),-1)
        
        out = self.network(x)
        
        if len(batch_shape)>1:
            out = out.reshape(*batch_shape,*out.shape[1:])
        
        return ModelOutput(reconstruction = self.out_process(out))
    
class LabelDecoder(BaseDecoder):
    
    def __init__(self, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(latent_dim,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
            nn.Linear(128,128),nn.BatchNorm1d(128),nn.LeakyReLU(),
            nn.Linear(128,10), nn.Softmax()
        )

    def forward(self, z):
        return ModelOutput(reconstruction = self.layers(z))
    
# Top encoders

class TopEncoder(BaseEncoder):
    
    def __init__(self, input_dim, msg_dim):
        super().__init__()
        
        self.layers = nn.Linear(input_dim,msg_dim)
    def forward(self, z):
        return ModelOutput(embedding = self.layers(z), log_covariance = None)


# Joint encoder

class JointEncoder(BaseEncoder):
    
    def __init__(self, msg_dim, latent_dim):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(msg_dim,512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(msg_dim,512),nn.BatchNorm1d(512),nn.LeakyReLU()
        )
        self.fc_embedding = nn.Linear(512,latent_dim)
        self.fc_covariance = nn.Linear(512,latent_dim)
        
    def forward(self, z):
        h = self.layers(z)
        return ModelOutput(
            embedding = self.fc_embedding(h),
            log_covariance = self.fc_covariance(h)
        )

# Top decoders

class Topdecoder(BaseDecoder):
    
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, 512),nn.BatchNorm1d(512),nn.LeakyReLU(),
            nn.Linear(512, input_dim)
            

        )

    def forward(self, z):
        
        return ModelOutput(reconstruction = self.layers(z))


# Model config

model_config = NexusConfig(
    n_modalities=4,
    input_dims = dict(image = (1,28,28), audio = (1,32,28), trajectory = (200,), label=(10,)),
    modalities_specific_dim=dict(image = 64, audio = 128, label = 5, trajectory = 16),
    latent_dim=32,
    msg_dim=512,
    gammas= dict(image = 1.0, trajectory = 50.0, audio = 1.0, label= 50.0),
    bottom_betas=dict(image = 1.0,trajectory = 1.0,audio = 1.0,label = 1.0),
    uses_likelihood_rescaling=True,
    rescale_factors=dict(image = 1.0, trajectory=50.0, audio = 1.0, label=50.0),
    top_beta=1.0,
    warmup = 20,
    dropout_rate=0.2)

model = Nexus(model_config,
              encoders = dict(image = ImageEncoder((1,28, 28), model_config.modalities_specific_dim['image']),
                              audio = SoundEncoder(model_config.modalities_specific_dim['audio']),
                              trajectory = TrajectoryEncoder(200,layer_sizes=[512, 512, 512], output_dim=16),
                              label = LabelEncoder(5)
                              ),
              decoders=dict(image = ImageDecoder((1,28,28),model_config.modalities_specific_dim['image']),
                            audio = SoundDecoder(model_config.modalities_specific_dim['audio']),
                            trajectory = TrajectoryDecoder(16, [512, 512,512], 200),
                            label = LabelDecoder(5)
                            ),
              top_encoders=dict(image = TopEncoder(model_config.modalities_specific_dim['image'],model_config.msg_dim),
                                audio = TopEncoder(model_config.modalities_specific_dim['audio'],model_config.msg_dim),
                                trajectory=TopEncoder(model_config.modalities_specific_dim['trajectory'],model_config.msg_dim),
                                label=TopEncoder(model_config.modalities_specific_dim['label'],model_config.msg_dim),
                                ),
              
              top_decoders =dict(image = Topdecoder(model_config.modalities_specific_dim['image'],model_config.latent_dim),
                                audio = Topdecoder(model_config.modalities_specific_dim['audio'],model_config.latent_dim),
                                trajectory=Topdecoder(model_config.modalities_specific_dim['trajectory'],model_config.latent_dim),
                                label=Topdecoder(model_config.modalities_specific_dim['label'],model_config.latent_dim),
                                ),
              
              joint_encoder=JointEncoder(model_config.msg_dim, model_config.latent_dim)
              
              )


# print(model(next(iter(DataLoader(train_set,2)))))


########## Training #######
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig

training_config = BaseTrainerConfig(
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=100,
    learning_rate=1e-3,
    output_dir=f"../reproduce_nexus",
    steps_predict=5,
    optimizer_cls="Adam"
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_nexus")

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    eval_dataset=None,
    training_config=training_config,
    callbacks=callbacks,
)

trainer.train()

trainer._best_model.push_to_hf_hub(
    f"asenella/reproduce_nexus"
)
