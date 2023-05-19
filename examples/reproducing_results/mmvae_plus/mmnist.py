import numpy as np
import torch
from multivae.data.datasets.mmnist import MMNISTDataset
from torch import nn
from multivae.models.base.base_model import BaseEncoder, BaseDecoder, ModelOutput
import torch.nn.functional as F

from multivae.trainers.base.callbacks import ProgressBarCallback, WandbCallback

##### Architectures #####


def actvn(x):
    out = torch.nn.functional.leaky_relu(x, 2e-1)
    return out


class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


# Classes
class Enc(BaseEncoder):
    """ Generate latent parameters for SVHN image data. """

    def __init__(self, ndim_w, ndim_u):
        super().__init__()
        self.latent_dim = ndim_u
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 1024  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks_w = [
            ResnetBlock(nf, nf)
        ]

        blocks_u = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks_w += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]
            blocks_u += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img_w = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet_w = nn.Sequential(*blocks_w)
        self.fc_mu_w = nn.Linear(self.nf0*s0*s0, ndim_w)
        self.fc_lv_w = nn.Linear(self.nf0*s0*s0, ndim_w)

        self.conv_img_u = nn.Conv2d(3, 1 * nf, 3, padding=1)
        self.resnet_u = nn.Sequential(*blocks_u)
        self.fc_mu_u = nn.Linear(self.nf0 * s0 * s0, ndim_u)
        self.fc_lv_u = nn.Linear(self.nf0 * s0 * s0, ndim_u)

    def forward(self, x):
        # batch_size = x.size(0)
        out_w = self.conv_img_w(x)
        out_w = self.resnet_w(out_w)
        out_w = out_w.view(out_w.size()[0], self.nf0*self.s0*self.s0)
        lv_w = self.fc_lv_w(out_w)

        out_u = self.conv_img_u(x)
        out_u = self.resnet_u(out_u)
        out_u = out_u.view(out_u.size()[0], self.nf0 * self.s0 * self.s0)
        lv_u = self.fc_lv_u(out_u)
        
        output = ModelOutput(
            embedding = self.fc_mu_u(out_u),
            style_embedding = self.fc_mu_w(out_w),
            log_covariance = lv_u,
            style_log_covariance = lv_w
        )

        return output

class Dec(BaseDecoder):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, ndim):
        super().__init__()

        # NOTE: I've set below variables according to Kieran's suggestions
        s0 = self.s0 = 7  # kwargs['s0']
        nf = self.nf = 64  # nfilter
        nf_max = self.nf_max = 512  # nfilter_max
        size = 28

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(ndim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        out = self.fc(z).view(-1, self.nf0, self.s0, self.s0)
        out = self.resnet(out)
        out = self.conv_img(actvn(out))

        #if len(z.size()) == 2:
        #out = out.view(*z.size()[:1], *out.size()[1:]).unsqueeze(0)
        #else:
        out = out.view(*z.size()[:2], *out.size()[1:])

        # consider also predicting the length scale
        return ModelOutput(reconstruction = out)
    
    
###### Model Config ########
from multivae.models.mmvaePlus import MMVAEPlusConfig, MMVAEPlus

modalities = ['m0','m1','m2','m3','m4']

model_config = MMVAEPlusConfig(
    n_modalities=5,
    K=10,
    decoders_dist={m : 'laplace' for m in modalities},
    decoder_dist_params= {m : dict(scale = 0.75) for m in modalities},
    prior_and_posterior_dist='laplace_with_softmax',
    beta=1,
    modalities_specific_dim=20,
    latent_dim=20,
    input_dims={m : (3,28,28) for m in modalities},
    learn_priors=False,
    
    
)

encoders = { m : Enc(model_config.modalities_specific_dim,ndim_u=model_config.latent_dim) for m in modalities}
decoders = { m : Dec(model_config.latent_dim + model_config.modalities_specific_dim) for m in modalities}

model = MMVAEPlus(model_config, encoders, decoders)


######## Dataset #########

train_data = MMNISTDataset(data_path='../data/MMNIST', split='train', download=True)
# test_data = MMNISTDataset(data_path='../data/MMNIST', split='test')


########## Training #######
from multivae.trainers.base import BaseTrainer, BaseTrainerConfig

training_config = BaseTrainerConfig(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_epochs=50,
    learning_rate=1e-3,
    start_keep_best_epoch=51,
    output_dir=f'../reproduce_mmvaep'
    
    
)

# Set up callbacks
wandb_cb = WandbCallback()
wandb_cb.setup(training_config, model_config, project_name="reproducing_mmvae")

callbacks = [ProgressBarCallback(), wandb_cb]

trainer = BaseTrainer(model,
                      train_data,
                    #   test_data,
                      training_config,
                      callbacks)

trainer.train()