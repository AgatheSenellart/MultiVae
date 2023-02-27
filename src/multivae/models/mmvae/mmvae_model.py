import torch
import numpy as np
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
            

        self.prior_mean = torch.zeros((self.latent_dim,))
        self.prior_std = torch.ones((self.latent_dim,))

        if model_config.learn_prior:
            self.prior_mean.requires_grad()
            self.prior_std.requires_grad()

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        # First compute all the encodings for all modalities
        embeddings = {}
        qz_xs = {}
        reconstructions = {}

        for cond_mod in self.encoders:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            mu, log_var = output.embedding, output.log_covariance

            if self.model_config.posterior_dist == "laplace_with_softmax":
                sigma = torch.softmax(log_var, dim=-1)
            else:
                sigma = torch.exp(0.5 * log_var)

            z_x = self.post_dist(mu, sigma).rsample([self.K])
            # The DREG loss uses detached parameters in the loss computation afterwards.
            qz_x = self.post_dist(mu.detach(), sigma.detach())

            # Then compute all the cross-modal reconstructions
            reconstructions[cond_mod] = {}
            for recon_mod in embeddings:
                decoder = self.decoders[recon_mod]
                recon = decoder(z_x)["reconstruction"]
                reconstructions[cond_mod][recon_mod] = recon

            qz_xs[cond_mod] = qz_x
            embeddings[cond_mod] = z_x

        # Compute DREG loss
        for mod in embeddings:
            z = embeddings[mod]
            prior = self.prior_dist(self.prior_mean, self.prior_std)
            lpz = prior.log_prob(z).sum(-1)
            lqz_x = torch.stack([qz_x[m].log_prob(z).sum(-1) for m in qz_xs])
            lqz_x = torch.logsumexp(lqz_x, dim=0) - np.log(self.n_modalities)
            lpx_z = reconstructions

        return

    def _m_dreg_looser(
        self,
        embeddings,
        qz_xs,
    ):
        """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
        This version is the looser bound---with the average over modalities outside the log
        """
        qz_xs, px_zs, zss, qz_x_params = model(x, K)
        qz_xs_ = [
            model.qz_x(*[p.detach() for p in qz_x_params[i]])
            for i in range(len(model.vaes))
        ]
        lws = []
        for r, vae in enumerate(model.vaes):
            lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
            lqz_x = log_mean_exp(
                torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_])
            )
            lpx_z = [
                px_z.log_prob(x[d])
                .view(*px_z.batch_shape[:2], -1)
                .mul(model.lik_scaling[d])
                .sum(-1)
                for d, px_z in enumerate(px_zs[r])
            ]
            lpx_z = torch.stack(lpx_z).sum(0)
            lw = lpz + lpx_z - lqz_x
            lws.append(lw)
        return torch.stack(lws), torch.stack(zss)

    def m_dreg_looser(model, x, K, epoch, warmup, beta_prior):
        """Computes dreg estimate for log p_\theta(x) for multi-modal vae
        This version is the looser bound---with the average over modalities outside the log
        """
        S = compute_microbatch_split(x, K)
        x_split = zip(*[_x.split(S) for _x in x])
        lw, zss = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
        lw = torch.cat(lw, 2)  # concat on batch
        zss = torch.cat(zss, 2)  # concat on batch
        with torch.no_grad():
            grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
            if zss.requires_grad:
                zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
        return (grad_wt * lw).mean(0).sum(), {}
