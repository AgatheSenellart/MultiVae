from typing import Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from ..joint_models import BaseJointModel
from .jmvae_config import JMVAEConfig


class JMVAE(BaseJointModel):

    """The Joint Multimodal Variational Autoencoder model.

    Args:
        model_config (JMVAEConfig): An instance of JMVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoder (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.

        joint_encoder (~pythae.models.nn.base_architectures.BaseEncoder) : An instance of
            BaseEncoder that takes all the modalities as an input. If none is provided, one is
            created from the unimodal encoders. Default : None.
    """

    def __init__(
        self,
        model_config: JMVAEConfig,
        encoders: dict = None,
        decoders: dict = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        self.model_name = "JMVAE"

        self.alpha = model_config.alpha
        self.warmup = model_config.warmup

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ):
        """Compute encodings from the posterior distributions conditioning on all the modalities or
        a subset of the modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to encode.
            cond_mod (Union[list, str], optional): The modalities to use to compute the posterior
            distribution. Defaults to 'all'.
            N (int, optional): The number of samples to generate from the posterior distribution
            for each datapoint. Defaults to 1.

        Raises:
            AttributeError: _description_
            AttributeError: _description_

        Returns:
            ModelOutput: Containing z the latent variables.
        """
        self.eval()

        mcmc_steps = kwargs.pop("mcmc_steps", 100)
        n_lf = kwargs.pop("n_lf", 10)
        eps_lf = kwargs.pop("eps_lf", 0.01)

        if cond_mod == "all" or (
            type(cond_mod) == list and len(cond_mod) == self.n_modalities
        ):
            output = self.joint_encoder(inputs.data)
            sample_shape = [] if N == 1 else [N]
            z = dist.Normal(
                output.embedding, torch.exp(0.5 * output.log_covariance)
            ).rsample(sample_shape)
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)

        if type(cond_mod) == list and len(cond_mod) != 1:
            z = self.sample_from_poe_subset(
                cond_mod,
                inputs.data,
                ax=None,
                mcmc_steps=mcmc_steps,
                n_lf=n_lf,
                eps_lf=eps_lf,
                K=N,
                divide_prior=True,
            )
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)

        elif type(cond_mod) == list and len(cond_mod) == 1:
            cond_mod = cond_mod[0]
        if cond_mod in self.modalities_name:
            output = self.encoders[cond_mod](inputs.data[cond_mod])
            sample_shape = [] if N == 1 else [N]

            z = dist.Normal(
                output.embedding, torch.exp(0.5 * output.log_covariance)
            ).rsample(
                sample_shape
            )  # shape N x len_data x latent_dim
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)
        else:
            raise AttributeError(
                f"Modality of name {cond_mod} not handled. The"
                f" modalities that can be encoded are {list(self.encoders.keys())}"
            )

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Performs a forward pass of the JMVAE model on inputs.

        Args:
            inputs (MultimodalBaseDataset)
            warmup (int) : number of warmup epochs to do. The weigth of the regularization augments
                linearly to reach 1 at the end of the warmup. The enforces the optimization of
                the reconstruction term only at first.
            epoch (int) : the epoch number during which forward is called.

        Returns:
            ModelOutput
        """

        epoch = kwargs.pop("epoch", 1)

        # Compute the reconstruction term
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        sigma = torch.exp(0.5 * log_var)
        qz_xy = dist.Normal(mu, sigma)
        z_joint = qz_xy.rsample()

        recon_loss = 0

        # Decode in each modality
        len_batch = 0
        for mod in self.decoders:
            x_mod = inputs.data[mod]
            len_batch = len(x_mod)
            recon_mod = self.decoders[mod](z_joint).reconstruction
            recon_loss += (
                -self.recon_log_probs[mod](recon_mod, x_mod) * self.rescale_factors[mod]
            ).sum()

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Compute the KL between unimodal and joint encoders
        LJM = 0
        for mod in self.encoders:
            output = self.encoders[mod](inputs.data[mod])
            uni_mu, uni_log_var = output.embedding, output.log_covariance
            LJM += (
                1
                / 2
                * (
                    uni_log_var
                    - log_var
                    + (torch.exp(log_var) + (mu - uni_mu) ** 2) / torch.exp(uni_log_var)
                    - 1
                )
            )

        LJM = LJM.sum() * self.alpha

        # Compute the total loss to minimize

        reg_loss = KLD + LJM
        if epoch >= self.warmup:
            beta = 1
        else:
            beta = epoch / self.warmup
        recon_loss, reg_loss = recon_loss / len_batch, reg_loss / len_batch
        elbo = (recon_loss + KLD)/len_batch
        loss = recon_loss + beta * reg_loss

        metrics = dict(loss_no_ponderation=reg_loss + recon_loss, beta=beta,elbo=elbo)

        output = ModelOutput(loss=loss, metrics=metrics)

        return output

    def sample_from_moe_subset(self, subset: list, data: dict):
        """Sample z from the mixture of posteriors from the subset.
        Torch no grad is activated, so that no gradient are computed during the forward pass of the encoders.

        Args:
            subset (list): the modalities to condition on
            data (list): The data
            K (int) : the number of samples per datapoint
        """
        # Choose randomly one modality for each sample
        n_batch = len(data[list(data.keys())[0]])

        indices = np.random.choice(subset, size=n_batch)
        zs = torch.zeros((n_batch, self.latent_dim)).to(
            data[list(data.keys())[0]].device
        )

        for m in subset:
            with torch.no_grad():
                encoder_output = self.encoders[m](data[m][indices == m])
                mu, log_var = encoder_output.embedding, encoder_output.log_covariance
                zs[indices == m] = dist.Normal(mu, torch.exp(0.5 * log_var)).rsample()
        return zs

    def compute_poe_posterior(
        self, subset: list, z_: torch.Tensor, data: list, divide_prior=True, grad=True
    ):
        """Compute the log density of the product of experts for Hamiltonian sampling.

        Args:
            subset (list): the modalities of the poe posterior
            z_ (torch.Tensor): the latent variables (len(data[0]), latent_dim)
            data (list): _description_
            divide_prior (bool) : wether or not to divide by the prior

        Returns:
            tuple : likelihood and gradients
        """

        lnqzs = 0

        z = z_.clone().detach().requires_grad_(grad)

        if divide_prior:
            # print('Dividing by the prior')
            lnqzs += (0.5 * (torch.pow(z, 2) + np.log(2 * np.pi))).sum(dim=1)

        for m in subset:
            # Compute lnqz

            vae_output = self.encoders[m](data[m])
            mu, log_var = vae_output.embedding, vae_output.log_covariance
            z0 = dist.Normal(mu, torch.exp(0.5 * log_var)).rsample()

            log_q_z0 = (
                -0.5
                * (
                    log_var
                    + np.log(2 * np.pi)
                    + torch.pow(z0 - mu, 2) / torch.exp(log_var)
                )
            ).sum(dim=1)
            lnqzs += log_q_z0  # n_data_points x 1

        if grad:
            g = torch.autograd.grad(lnqzs.sum(), z)[0]
            return lnqzs, g
        else:
            return lnqzs

    def sample_from_poe_subset(
        self,
        subset,
        data,
        ax=None,
        mcmc_steps=100,
        n_lf=10,
        eps_lf=0.01,
        K=1,
        divide_prior=True,
    ):
        """Sample from the product of experts using Hamiltonian sampling.

        Args:
            subset (List[int]):
            gen_mod (int):
            data (dict or MultimodalDataset):
            K (int, optional): . Defaults to 100.
        """

        # Multiply the data to have multiple samples per datapoints
        n_data = len(data[list(data.keys())[0]])
        data = {d: torch.cat([data[d]] * K) for d in data}
        device = data[list(data.keys())[0]].device

        n_samples = len(data[list(data.keys())[0]])
        acc_nbr = torch.zeros(n_samples, 1).to(device)

        # First we need to sample an initial point from the mixture of experts
        z0 = self.sample_from_moe_subset(subset, data)
        z = z0

        # fig, ax = plt.subplots()
        pos = []
        grad = []
        for i in range(mcmc_steps):
            pos.append(z[0].detach().cpu())

            # print(i)
            gamma = torch.randn_like(z, device=device)
            rho = gamma  # / self.beta_zero_sqrt

            # Compute ln q(z|X_s)
            ln_q_zxs, g = self.compute_poe_posterior(
                subset, z, data, divide_prior=divide_prior
            )

            grad.append(g[0].detach().cpu())

            H0 = -ln_q_zxs + 0.5 * torch.norm(rho, dim=1) ** 2
            # print(H0)
            # print(model.G_inv(z).det())
            for k in range(n_lf):
                # z = z.clone().detach().requires_grad_(True)
                # log_det = G(z).det().log()

                # g = torch.zeros(n_samples, model.latent_dim).cuda()
                # for i in range(n_samples):
                #    g[0] = -grad(log_det, z)[0][0]

                # step 1
                rho_ = rho - (eps_lf / 2) * (-g)

                # step 2
                z = z + eps_lf * rho_

                # z_ = z_.clone().detach().requires_grad_(True)
                # log_det = 0.5 * G(z).det().log()
                # log_det = G(z_).det().log()

                # g = torch.zeros(n_samples, model.latent_dim).cuda()
                # for i in range(n_samples):
                #    g[0] = -grad(log_det, z_)[0][0]

                # Compute the updated gradient
                ln_q_zxs, g = self.compute_poe_posterior(subset, z, data, divide_prior)

                # print(g)
                # g = (Sigma_inv @ (z - mu).T).reshape(n_samples, 2)

                # step 3
                rho__ = rho_ - (eps_lf / 2) * (-g)

                # tempering
                beta_sqrt = 1

                rho = rho__
                # beta_sqrt_old = beta_sqrt

            H = -ln_q_zxs + 0.5 * torch.norm(rho, dim=1) ** 2
            # print(H, H0)

            alpha = torch.exp(H0 - H)
            # print(alpha)

            # print(-log_pi(best_model, z, best_model.G), 0.5 * torch.norm(rho, dim=1) ** 2)
            acc = torch.rand(n_samples).to(device)
            moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)

            acc_nbr += moves

            z = z * moves + (1 - moves) * z0
            z0 = z

        pos = torch.stack(pos)
        grad = torch.stack(grad)

        sh = (n_data, self.latent_dim) if K == 1 else (K, n_data, self.latent_dim)
        z = z.detach().resize(*sh)
        return z.detach()


    def compute_joint_nll_paper(
            self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
        ):
            """
            Return the estimated negative log-likelihood summed over the input batch.
            The negative log-likelihood is estimated using importance sampling.

            Args :
                inputs : the data to compute the joint likelihood"""

            # First compute all the parameters of the joint posterior q(z|x,y)

            print(
                "Started computing the negative log_likelihood on inputs. This function"
                " can take quite a long time to run."
            )

            joint_output = self.joint_encoder(inputs.data)
            mu, log_var = joint_output.embedding, joint_output.log_covariance

            sigma = torch.exp(0.5 * log_var)
            qz_xy = dist.Normal(mu, sigma)
            # And sample from the posterior
            z_joint = qz_xy.rsample([K])  # shape K x n_data x latent_dim
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
                    for mod in ['images']:
                        decoder = self.decoders[mod]
                        recon = decoder(latents)[
                            "reconstruction"
                        ]  # (batch_size_K, *decoder_output_shape)
                        x_m = inputs.data[mod][i]  # (*input_shape)
                        lpx_zs += self.recon_log_probs[mod](recon, x_m).reshape(recon.size(0),-1).sum(-1)

                    # Compute ln(p(z))
                    prior = dist.Normal(0, 1)
                    lpz = prior.log_prob(latents).sum(dim=-1)

                    # Compute posteriors -ln(q(z|x,y))
                    qz_xy = dist.Normal(mu[i], sigma[i])
                    lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                    ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                    lnpxs.append(ln_px)

                    # next batch
                    start_idx += batch_size_K
                    stop_idx = min(stop_idx + batch_size_K, K)

                ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

            return -ll 
