from pythae.models import VAE, VAEConfig, BaseAEConfig
from pythae.data import BaseDataset
from pythae.trainers import BaseTrainer,BaseTrainerConfig
import torch
from config import random_split
from architectures import Encoder_Conv_VAE_MNIST, Decoder_Conv_AE_MNIST, SoundEncoder, SoundDecoder, TrajectoryDecoder, TrajectoryEncoder

for modality in ['trajectory']:
    for classe in range(10):
        data = dict()

        (
                    data['label'],
                    data['image'],
                    data['trajectory'],
                    data['audio'],
                    _traj_normalization,
                    _audio_normalization,
                ) = torch.load('/home/asenella/scratch/data/MHD/mhd_train.pt')



        def unstack_tensor(tensor, dim=0):
            tensor_lst = []
            for i in range(tensor.size(dim)):
                tensor_lst.append(tensor[i])
            tensor_unstack = torch.cat(tensor_lst, dim=0)
            return tensor_unstack

        _a_data = data['audio'].permute(1,2,3,0)
        _a_data = unstack_tensor(_a_data,dim=0).unsqueeze(0)
        data['audio'] = _a_data.permute(3,0, 2,1)


        dataset = BaseDataset(data[modality][data['label']==classe], labels=data['label'][data['label']==classe])

        train,val = random_split(dataset,[0.85,0.15])

        if modality == 'image':
            encoder = Encoder_Conv_VAE_MNIST(BaseAEConfig((3,28,28), latent_dim = 64))
            decoder = Decoder_Conv_AE_MNIST(BaseAEConfig(latent_dim=64, input_dim=(3,28,28)))

        elif modality == 'audio':
            encoder = SoundEncoder(64)
            decoder = SoundDecoder(64)

        else :
            encoder = TrajectoryEncoder(200, layer_sizes=[512, 512, 512], output_dim=64)
            decoder=TrajectoryDecoder(64, [512,512,512],output_dim=200)
            
            
        model_config = VAEConfig(latent_dim=64)
        model=VAE(model_config,encoder, decoder)

        trainer_config = BaseTrainerConfig(
            num_epochs=100,
            output_dir=f'/home/asenella/scratch/mhd_unimodal_encoders/{modality}/{classe}'
        )
        trainer = BaseTrainer(
            model, train, val, trainer_config
        )

        trainer.train()