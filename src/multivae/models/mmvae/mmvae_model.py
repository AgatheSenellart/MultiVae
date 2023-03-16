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

    """
    Implements the MMVAE model from the paper : (Variational Mixture-of-Experts Autoencoders
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

        self.prior_mean.requires_grad_(model_config.learn_prior)
        self.prior_std.requires_grad_(model_config.learn_prior)

        self.model_name = "MMVAE"

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        # TODO : maybe implement a minibatch strategy for stashing the gradients before
        # backpropagation when using a large number k.
        # Also, I've only implemented the dreg_looser loss but it may be nice to offer other options.

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

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """Return the average estimated negative log-likelihood over the inputs.
        The negative log-likelihood is estimated using importance sampling.

        Args :
            inputs : the data to compute the joint likelihood"""

        print(
            "Started computing the negative log_likelihood on inputs. This function"
            " can take quite a long time to run."
        )

        # First compute all the parameters of the joint posterior q(z|x,y)
        qz_xs = []
        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance

            if self.model_config.prior_and_posterior_dist == "laplace_with_softmax":
                sigma = torch.softmax(log_var, dim=-1)
            else:
                sigma = torch.exp(0.5 * log_var)

            qz_xs.append(self.post_dist(mu, sigma))

        z_joint = self.encode(inputs, N=K).z
        z_joint = z_joint.permute(1, 0, 2)
        n_data, _, latent_dim = z_joint.shape

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            while start_idx < stop_idx:
                latents = z_joint[i][start_idx:stop_idx]

                # Compute p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0  # ln(p(x,y|z))
                for mod in inputs.data:
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, nb_channels, w, h)
                    x_m = inputs.data[mod][i]  # (nb_channels, w, h)

                    dim_reduce = tuple(range(1, len(recon.shape)))
                    lpx_zs += self.recon_log_probs[mod](recon, x_m).sum(dim=dim_reduce)

                # Compute ln(p(z))
                prior = self.prior_dist(self.prior_mean, self.prior_std)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|x,y))

                lqz_xy = torch.logsumexp(
                    torch.stack([q.log_prob(latents).sum(-1) for q in qz_xs]), dim=0
                ) - np.log(self.n_modalities)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll / n_data
