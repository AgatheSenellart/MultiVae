import logging
from typing import Dict, Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.normalizing_flows.base import BaseNF
from pythae.models.normalizing_flows.maf import MAF, MAFConfig
from torch.nn import ModuleDict

from ...data.datasets.base import MultimodalBaseDataset
from ..base.base_utils import rsample_from_gaussian
from ..joint_models import BaseJointModel
from ..nn.base_architectures import BaseJointEncoder
from .jnf_config import JNFConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class JNF(BaseJointModel):
    """The JNF model.

    Args:
        model_config (JNFConfig): Contains parameters for the JNF model.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary
            containing the modalities names and the encoders for each modality. Each encoder is
            an instance of Pythae's BaseEncoder.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary
            containing the modalities names and the decoders for each modality. Each decoder is an
            instance of Pythae's BaseDecoder.

        joint_encoder (~multivae.models.nn.base_architectures.BaseJointEncoder) : Takes all
            the modalities as an input. If none is provided, one is
            created from the unimodal encoders. Default : None.

        flows (Dict[str, ~pythae.models.normalizing_flows.BaseNF]) : A dictionary containing the
            modalities names and the flows to use for each modality. If None is provided, a
            default MAF flow is used for each modality.


    """

    def __init__(
        self,
        model_config: JNFConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        joint_encoder: Union[BaseJointEncoder, None] = None,
        flows: Dict[str, BaseNF] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        if flows is None:
            flows = self._default_flows(model_config)
        else:
            self.model_config.custom_architectures.append("flows")

        self._set_flows(flows)

        self.model_name = "JNF"
        self.warmup = model_config.warmup
        self.reset_optimizer_epochs = [self.warmup + 1]
        self.beta = model_config.beta

    def _default_flows(self, model_config):
        """Return default masked autoregressive flows for each modality."""
        flows = {}
        for modality in self.encoders:
            flows[modality] = MAF(MAFConfig(input_dim=(model_config.latent_dim,)))
        return flows

    def _set_flows(self, flows: Dict[str, BaseNF]):
        """Sanity check on the flows and set attribute."""
        if flows.keys() != self.encoders.keys():
            raise AttributeError(
                f"The keys of provided flows : {list(flows.keys())}"
                f" doesn't match the keys provided in encoders {list(self.encoders.keys())}"
                " or input_dims."
            )

        # Check that the flows are instances of BaseNF and that the input_dim for the
        # flows matches the latent_dimension
        self.flows = ModuleDict()
        for m in flows:
            if isinstance(flows[m], BaseNF) and flows[m].input_dim == (
                self.latent_dim,
            ):
                self.flows[m] = flows[m]
            else:
                raise AttributeError(
                    "The provided flows must be instances of the Pythae's BaseNF "
                    " class."
                )
        return

    def _set_torch_no_grad_on_joint_vae(self):
        # After the warmup, we freeze the architecture of the joint encoder and decoders
        self.joint_encoder.requires_grad_(False)
        self.decoders.requires_grad_(False)

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """Forward pass of the JNF model. Returns the loss and metrics."""
        # Check that the dataset is not incomplete
        super().forward(inputs)

        epoch = kwargs.pop("epoch", 1)

        # First compute the joint ELBO
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
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * self.beta

        if epoch <= self.warmup:
            return ModelOutput(
                # recon_loss=recon_loss / len_batch,
                # KLD=KLD / len_batch,
                loss=(recon_loss + KLD) / len_batch,
                loss_sum=recon_loss + KLD,
                metrics=dict(kld_prior=KLD, recon_loss=recon_loss / len_batch, ljm=0),
            )

        else:
            self._set_torch_no_grad_on_joint_vae()
            ljm = self._compute_ljm(inputs, z_joint)

            return ModelOutput(
                loss=ljm / len_batch,
                loss_sum=ljm,
                metrics=dict(
                    kld_prior=KLD,
                    recon_loss=recon_loss / len_batch,
                    ljm=ljm / len_batch,
                ),
            )

    def _compute_ljm(self, inputs, z_joint):
        """Compute the KL-divergence between unimodal posteriors and joint posterior.

        Args:
            inputs (MultimodalBaseDataset): the batch inputs
            z_joint (tensor): The batch joint representation computed from the joint encoder.
        """
        ljm = 0
        for mod in self.encoders:
            mod_output = self.encoders[mod](inputs.data[mod])
            mu0, log_var0 = mod_output.embedding, mod_output.log_covariance

            sigma0 = torch.exp(0.5 * log_var0)
            qz_x0 = dist.Normal(mu0, sigma0)

            # Compute -ln q_\phi_mod(z_joint|x_mod)
            flow_output = self.flows[mod](z_joint)
            z0 = flow_output.out

            ljm += -(qz_x0.log_prob(z0).sum(dim=-1) + flow_output.log_abs_det_jac).sum()
        return ljm

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        return_mean=False,
        **kwargs,
    ) -> ModelOutput:
        """Generate encodings conditioning on all modalities or a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The dataset to use for the conditional generation.
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities
                names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.
            return_mean (bool) : if True, returns the mean of the posterior distribution (instead of a sample).


        **kwargs:
            mcmc_steps(int) : the number of Monte-Carlo step to perform when sampling from the product
                of experts. Default to 100. If the coherences results are bad and the latent space is quite large,
                consider augmenting this number.
            n_lf (int) : The number of leapfrog steps in the Hamiltonian Monte Carlo Sampling.
                Default to 10.
            eps_lf (float) : the time step to use in the Hamiltonian Monte Carlo Sampling.
                default to 0.01.

        Returns:
            ModelOutput :
                Contains fields
                'z' (torch.Tensor (N, n_data, latent_dim))
                'one_latent_space' (bool) = True



        """
        mcmc_steps = kwargs.pop("mcmc_steps", 100)
        n_lf = kwargs.pop("n_lf", 10)
        eps_lf = kwargs.pop("eps_lf", 0.01)

        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        if len(cond_mod) == self.n_modalities:
            output = self.joint_encoder(inputs.data)

            z = rsample_from_gaussian(
                output.embedding, output.log_covariance, N, return_mean
            )

        elif len(cond_mod) != 1:
            z = self._sample_from_poe_subset(
                cond_mod,
                inputs.data,
                ax=None,
                mcmc_steps=mcmc_steps,
                n_lf=n_lf,
                eps_lf=eps_lf,
                K=N,
                divide_prior=True,
            )
            # no return mean option here

        elif len(cond_mod) == 1:
            cond_mod = cond_mod[0]
            output = self.encoders[cond_mod](inputs.data[cond_mod])

            z0 = rsample_from_gaussian(
                output.embedding, output.log_covariance, N, return_mean
            )

            flow_output = self.flows[cond_mod].inverse(
                z0.reshape(-1, self.latent_dim)
            )  # The reshaping is because MAF flows doesn't handle
            # any shape of input data (*,*input_dim)
            z = flow_output.out.reshape(z0.shape)

        else:
            raise AttributeError(
                f"Modality of name {cond_mod} not handled. The"
                f" modalities that can be encoded are {list(self.encoders.keys())}"
            )

        if N > 1 and kwargs.pop("flatten", False):
            N, l, d = z.shape
            z = z.reshape(l * N, d)
        return ModelOutput(z=z, one_latent_space=True)

    def _sample_from_moe_subset(self, subset: list, data: dict):
        """Sample z from the mixture of posteriors from the subset.
        Torch no grad is activated, so that no gradient are computed durin the forward pass of the encoders.

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

    def _compute_poe_posterior(
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
        with torch.set_grad_enabled(grad):
            lnqzs = 0

            z = z_.detach().clone().requires_grad_(grad)

            if divide_prior:
                lnqzs = lnqzs + (0.5 * (torch.pow(z, 2) + np.log(2 * np.pi))).sum(dim=1)
            for m in subset:
                # Compute lnqz
                flow_output = self.flows[m](z)
                vae_output = self.encoders[m](data[m])
                mu, log_var, z0 = (
                    vae_output.embedding,
                    vae_output.log_covariance,
                    flow_output.out,
                )

                log_q_z0 = (
                    -0.5
                    * (
                        log_var
                        + np.log(2 * np.pi)
                        + torch.pow(z0 - mu, 2) / torch.exp(log_var)
                    )
                ).sum(dim=1)
                lnqzs = (
                    lnqzs + log_q_z0 + flow_output.log_abs_det_jac
                )  # n_data_points x 1
            if grad:
                g = torch.autograd.grad(lnqzs.sum(), z)[0]
                return lnqzs, g
            else:
                return lnqzs

    def _sample_from_poe_subset(
        self,
        subset,
        data,
        ax=None,
        mcmc_steps=300,
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
        logger.info(
            "starting to sample from poe_subset, divide prior = %s", str(divide_prior)
        )

        # Multiply the data to have multiple samples per datapoints
        n_data = len(data[list(data.keys())[0]])
        data = {d: torch.cat([data[d]] * K) for d in data}
        device = data[list(data.keys())[0]].device

        n_samples = len(data[list(data.keys())[0]])
        acc_nbr = torch.zeros(n_samples, 1).to(device)

        # First we need to sample an initial point from the mixture of experts
        z0 = self._sample_from_moe_subset(subset, data)
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
            ln_q_zxs, g = self._compute_poe_posterior(
                subset, z, data, divide_prior=divide_prior
            )

            grad.append(g[0].detach().cpu())

            H0 = -ln_q_zxs + 0.5 * torch.norm(rho, dim=1) ** 2

            for k in range(n_lf):
                # step 1
                rho_ = rho - (eps_lf / 2) * (-g)

                # step 2
                z = z + eps_lf * rho_

                # Compute the updated gradient
                ln_q_zxs, g = self._compute_poe_posterior(subset, z, data, divide_prior)

                # step 3
                rho__ = rho_ - (eps_lf / 2) * (-g)

                # tempering
                beta_sqrt = 1

                rho = rho__
                # beta_sqrt_old = beta_sqrt

            H = -ln_q_zxs + 0.5 * torch.norm(rho, dim=1) ** 2

            alpha = torch.exp(H0 - H)

            acc = torch.rand(n_samples).to(device)
            moves = (acc < alpha).type(torch.int).reshape(n_samples, 1)

            acc_nbr += moves

            z = z * moves + (1 - moves) * z0
            z0 = z

        pos = torch.stack(pos)
        grad = torch.stack(grad)
        if ax is not None:
            ax.plot(pos[:, 0], pos[:, 1])
            ax.quiver(pos[:, 0], pos[:, 1], grad[:, 0], grad[:, 1])

        sh = (n_data, self.latent_dim) if K == 1 else (K, n_data, self.latent_dim)
        z = z.detach().resize(*sh)
        return z.detach()
