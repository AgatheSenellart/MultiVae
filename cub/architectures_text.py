import torch
from torch import nn
import torch.nn.functional as F
from multivae.models.base import BaseEncoder, BaseDecoder, ModelOutput
from torch.distributions import OneHotCategorical

class Constants(object):
    eta = 1e-20

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590


# Classes
class Enc(BaseEncoder):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latentDim_w, latentDim_z, dist):
        super(Enc, self).__init__()
        self.latent_dim=latentDim_z
        self.dist = dist
        self.embedding = nn.Linear(vocabSize, embeddingDim)
        self.enc_w = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True)
        )
        self.enc_z = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 8, fBase * 16, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1_w = nn.Linear(fBase * 16 * 16, latentDim_w)
        self.c2_w = nn.Linear(fBase * 16 * 16, latentDim_w)

        self.c1_z = nn.Conv2d(fBase * 16, latentDim_z, 4, 1, 0, bias=True)
        self.c2_z = nn.Conv2d(fBase * 16, latentDim_z, 4, 1, 0, bias=True)

        # self.ll_c1_z = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_z)])
        # self.ll_c2_z = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_z)])
        # self.ll_c1_w = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_w)])
        # self.ll_c2_w = nn.Sequential(*[nn.ReLU(), nn.Linear(128, latentDim_w)])

    def forward(self, x):
        
        x = x['one_hot']
        
        x_emb = self.embedding(x).unsqueeze(1)
        e_w = self.enc_w(x_emb)
        e_w = e_w.view(-1, fBase * 16 * 16)
        mu_w, lv_w = self.c1_w(e_w), self.c2_w(e_w)
        e_z = self.enc_z(x_emb)
        
        if x.shape[0] == 1:
            mu_z, lv_z = self.c1_z(e_z).squeeze().unsqueeze(0), self.c2_z(e_z).squeeze().unsqueeze(0)
        else :
            mu_z, lv_z = self.c1_z(e_z).squeeze(), self.c2_z(e_z).squeeze()
        
        if self.dist == 'Normal':
            return ModelOutput(
                embedding = mu_z,
                style_embedding = mu_w,
                log_covariance =  F.softplus(lv_z) + Constants.eta,
                style_log_covariance =F.softplus(lv_w) + Constants.eta)
                
                
        else:
            return ModelOutput(
                embedding = mu_z,
                style_embedding = mu_w,
                log_covariance = F.softmax(lv_z, dim=-1) * lv_w.size(-1) + Constants.eta,
                style_log_covariance = F.softmax(lv_w, dim=-1) * lv_w.size(-1) + Constants.eta
            )
        
       


class Dec(BaseDecoder):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latentDim_w, latentDim_z):
        
        super(Dec, self).__init__()
        self.joint_decoder_input_dim = fBase*8 if latentDim_w is not 0 else fBase*4
        
        self.latent_dim = latentDim_z
        self.dec_w = nn.Sequential(
            nn.ConvTranspose2d(latentDim_w, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
        )
        self.dec_z = nn.Sequential(
            nn.ConvTranspose2d(latentDim_z, fBase * 16, 4, 1, 0, bias=True),
            nn.BatchNorm2d(fBase * 16),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 16, fBase * 8, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 8, fBase * 8, 3, 1, 1, bias=True),
            nn.BatchNorm2d(fBase * 8),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 8, fBase * 4, (1, 4), (1, 2), (0, 1), bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
        )
        self.dec_h = nn.Sequential(
            nn.ConvTranspose2d(self.joint_decoder_input_dim, fBase * 4, 3, 1, 1, bias=True),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=True),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

        self.latent_dim_w = latentDim_w
        self.latent_dim_z = latentDim_z

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, u):
        # z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        
        if self.latent_dim_w ==0 :
            z = u
            z = z.unsqueeze(-1).unsqueeze(-1)
            hz = self.dec_z(z.view(-1, *z.size()[-3:]))
            out = self.dec_h(hz)
        else :
            w, z = torch.split(u, [self.latent_dim_w, self.latent_dim_z], dim=-1)
            z = z.unsqueeze(-1).unsqueeze(-1)
            hz = self.dec_z(z.view(-1, *z.size()[-3:]))
            w = w.unsqueeze(-1).unsqueeze(-1)
            hw = self.dec_w(w.view(-1, *w.size()[-3:]))
            h = torch.cat((hw, hz), dim=1)
            out = self.dec_h(h)
        out = out.view(*z.size()[:-3], *out.size()[1:]).view(-1, embeddingDim)
        # The softmax is key for this to work
        ret = self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize)
        return ModelOutput(reconstruction  = ret)
    
    
    
    
    