from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from torch.distributions import Laplace, Normal

from multivae.data.datasets.base import MultimodalBaseDataset

from ..base import BaseMultiVAE
from .mmvae_config import MMVAEConfig


class MMVAE(BaseMultiVAE):

    """Implements the MMVAE model from the paper : (Variational Mixture-of-Experts Autoencoders
    for Multi-Modal Deep Generative Models, Shi et al 2019,
    https://proceedings.neurips.cc/paper/2019/hash/0ae775a8cb3b499ad1fca944e6f5c836-Abstract.html)


    """

    def __init__(
        self, model_config: MMVAEConfig, encoders: dict = None, decoders: dict = None
    ):
        super().__init__(model_config, encoders, decoders)

        self.K = model_config.K

        if model_config.prior_and_posterior_dist == "laplace_with_softmax":
            self.post_dist = Laplace
            self.prior_dist = Laplace
        elif model_config.posterior_dist == "normal":
            self.post_dist = Normal
            self.prior_dist = Normal
        else:
            raise AttributeError(
                " The posterior_dist parameter must be "
                " either 'laplace_with_softmax' or 'normal'. "
                f" {model_config.posterior_dist} was provided."
            )

        self.prior_mean = torch.nn.Parameter(torch.zeros((self.latent_dim,)))
        self.prior_std = torch.nn.Parameter(torch.ones((self.latent_dim,)))

        if model_config.learn_prior:
            self.prior_mean.requires_grad_()
            self.prior_std.requires_grad_()

        self.model_name = "MMVAE"

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        # First compute all the encodings for all modalities
        embeddings = {}
        qz_xs = {}
        reconstructions = {}
        n_batch = len(list(inputs.data.values())[0])

        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance

            if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
                sigma = torch.softmax(log_var, dim=-1)
            else:
                sigma = torch.exp(0.5 * log_var)

            z_x = self.post_dist(mu, sigma).rsample([self.K])
            # The DREG loss uses detached parameters in the loss computation afterwards.
            qz_x = self.post_dist(mu.detach(), sigma.detach())

            # Then compute all the cross-modal reconstructions
            reconstructions[cond_mod] = {}
            for recon_mod in self.decoders:
                decoder = self.decoders[recon_mod]
                recon = decoder(z_x)["reconstruction"]
                reconstructions[cond_mod][recon_mod] = recon

            qz_xs[cond_mod] = qz_x
            embeddings[cond_mod] = z_x

        # Compute DREG loss
        output = self.dreg_looser(qz_xs, embeddings, reconstructions, inputs)
        return output

    def dreg_looser(self, qz_xs, embeddings, reconstructions, inputs):
        lw = []
        zss = []
        for mod in embeddings:
            z = embeddings[mod]  # (K, n_batch, latent_dim)

            prior = self.prior_dist(self.prior_mean, self.prior_std)
            lpz = prior.log_prob(z).sum(-1)
            lqz_x = torch.stack([qz_xs[m].log_prob(z).sum(-1) for m in qz_xs])
            lqz_x = torch.logsumexp(lqz_x, dim=0) - np.log(self.n_modalities)
            lpx_z = 0
            for recon_mod in reconstructions[mod]:
                x_recon = reconstructions[mod][recon_mod]
                K, n_batch = x_recon.shape[0], x_recon.shape[1]
                lpx_z -= (
                    self.recon_losses[recon_mod](
                        x_recon, torch.stack([inputs.data[recon_mod]] * K)
                    )
                    .reshape(K, n_batch, -1)
                    .sum(-1)
                    * self.rescale_factors[recon_mod]
                )
            loss = lpx_z + lpz - lqz_x
            lw.append(loss)
            zss.append(z)

        lw = torch.stack(lw)  # (n_modalities, K, n_batch)
        zss = torch.stack(zss)  # (n_modalities, K, n_batch,latent_dim)
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if zss.requires_grad:  # True except when we are in eval mode
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)

        lw = (lw * grad_wt).mean(0).sum()

        return ModelOutput(loss=-lw, metrics=dict())

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        # If the input cond_mod is a string : convert it to a list
        if type(cond_mod) == str:
            if cond_mod == "all":
                cond_mod = list(self.encoders.keys())
            elif cond_mod in self.encoders.keys():
                cond_mod = [cond_mod]
            else:
                raise AttributeError(
                    'If cond_mod is a string, it must either be "all" or a modality name'
                    f" The provided string {cond_mod} is neither."
                )

        if all([s in self.encoders.keys() for s in cond_mod]):
            # Choose one of the conditioning modalities at random
            mod = np.random.choice(cond_mod)
            print(mod)

            output = self.encoders[mod](inputs.data[mod])

            mu, log_var = output.embedding, output.log_covariance

            if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
                sigma = torch.softmax(log_var, dim=-1)
            else:
                sigma = torch.exp(0.5 * log_var)

            qz_x = self.post_dist(mu, sigma)
            sample_shape = torch.Size([]) if N == 1 else torch.Size([N])
            z = qz_x.rsample(sample_shape)

            flatten = kwargs.pop("flatten", False)
            if flatten:
                z = z.reshape(-1, self.latent_dim)

            return ModelOutput(z=z, one_latent_space=True)
