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

from multivae.models.nn.default_architectures import (
    BaseDictEncoders,
    MultipleHeadJointEncoder,
)

from ...data.datasets.base import MultimodalBaseDataset
from ..dcca import DCCA, DCCAConfig
from ..joint_models import BaseJointModel
from ..nn.default_architectures import BaseDictEncoders
from .jnf_dcca_config import JNFDccaConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class JNFDcca(BaseJointModel):

    """
    The JNFDcca model.

    Args:

        model_config (JNFDccaConfig): An instance of JNFConfig in which any model's parameters is
            made available.

        encoders (Dict[str,BaseEncoder]): A dictionary containing the modalities names and the encoders for each
            modality. Each encoder is an instance of Pythae's BaseEncoder.

        decoders (Dict[str,BaseDecoder]): A dictionary containing the modalities names and the decoders for each
            modality. Each decoder is an instance of Pythae's BaseDecoder.

        joint_encoder (BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.

        flows (Dict[str,BaseNF]) : A dictionary containing the modalities names and the flows to use for
            each modality. If None is provided, a default MAF flow is used for each modality.

        dcca_networks (Dict[str, BaseEncoder]) : A dictionary containing the networks to use in the DCCA module
            for each modality. If None is provided, default MLPs are used.

    """

    def __init__(
        self,
        model_config: JNFDccaConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        flows: Dict[str, BaseNF] = None,
        dcca_networks: Dict[str, BaseEncoder] = None,
        **kwargs,
    ):
        self.dcca_config = DCCAConfig(
            n_modalities=model_config.n_modalities,
            embedding_dim=model_config.embedding_dcca_dim,
            use_all_singular_values=model_config.use_all_singular_values,
        )

        if dcca_networks is None:
            if model_config.input_dims is None:
                raise AttributeError(
                    "Please provide dcca_networks or input_dims for the modalities in the model_config."
                    " The model can't be initialized if both are missing."
                )
            else:
                if len(model_config.input_dims.keys()) != model_config.n_modalities:
                    raise AttributeError(
                        f"The provided number of input_dims {len(model_config.input_dims.keys())} doesn't"
                        f"match the number of modalities ({model_config.n_modalities} in model config "
                    )
                dcca_networks = BaseDictEncoders(
                    model_config.input_dims, model_config.embedding_dcca_dim
                )
        else:
            model_config.custom_architectures.append("dcca_networks")

        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        # The default encoders for this model have (embedding_dcca_dim, ) as input_size
        if encoders is None:
            encoders_input_dims = {
                k: (model_config.embedding_dcca_dim,) for k in dcca_networks
            }
            self.set_encoders(
                BaseDictEncoders(encoders_input_dims, model_config.latent_dim)
            )

        # The default joint_encoder for this model is engineered from the DCCA networks and
        # not from the encoders
        if joint_encoder is None:
            # Create a MultiHead Joint Encoder MLP
            self.set_joint_encoder(
                MultipleHeadJointEncoder(dcca_networks, model_config)
            )

        self.DCCA_module = DCCA(self.dcca_config, dcca_networks)
        self.dcca_networks = self.DCCA_module.networks

        if flows is None:
            flows = dict()
            for modality in self.encoders:
                flows[modality] = MAF(MAFConfig(input_dim=(model_config.latent_dim,)))
        else:
            model_config.custom_architectures.append("flows")

        self.set_flows(flows)
        self.model_name = "JNFDcca"
        self.warmup = model_config.warmup
        self.nb_epochs_dcca = model_config.nb_epochs_dcca
        self.reset_optimizer_epochs = [
            self.nb_epochs_dcca,
            self.nb_epochs_dcca + self.warmup,
        ]

    def set_flows(self, flows: Dict[str, BaseNF]):
        # check that the keys corresponds with the encoders keys
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
            if isinstance(flows[m], BaseNF):
                if flows[m].input_dim == (self.latent_dim,):
                    self.flows[m] = flows[m]
                else:
                    raise AttributeError(
                        "The provided flows don't have the right input dim."
                    )
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

    def _set_torch_no_grad_on_dcca_module(self):
        self.DCCA_module.requires_grad_(False)

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        epoch = kwargs.pop("epoch", 1)

        if epoch <= self.nb_epochs_dcca:
            return self.DCCA_module(inputs)
        else:
            self._set_torch_no_grad_on_dcca_module()

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
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if epoch <= self.warmup + self.nb_epochs_dcca:
            return ModelOutput(
                recon_loss=recon_loss / len_batch,
                KLD=KLD / len_batch,
                loss=(recon_loss + KLD) / len_batch,
                metrics=dict(kld_prior=KLD, recon_loss=recon_loss / len_batch, ljm=0),
            )

        else:
            self._set_torch_no_grad_on_joint_vae()
            ljm = 0
            for mod in self.encoders:
                dcca_embed = self.DCCA_module.networks[mod](inputs.data[mod]).embedding
                mod_output = self.encoders[mod](dcca_embed)
                mu0, log_var0 = mod_output.embedding, mod_output.log_covariance

                sigma0 = torch.exp(0.5 * log_var0)
                qz_x0 = dist.Normal(mu0, sigma0)

                # Compute -ln q_\phi_mod(z_joint|x_mod)
                flow_output = self.flows[mod](z_joint)
                z0 = flow_output.out

                ljm += -(
                    qz_x0.log_prob(z0).sum(dim=-1) + flow_output.log_abs_det_jac
                ).sum()

            return ModelOutput(
                recon_loss=recon_loss / len_batch,
                KLD=KLD / len_batch,
                loss=ljm / len_batch,
                ljm=ljm / len_batch,
                metrics=dict(
                    kld_prior=KLD,
                    recon_loss=recon_loss / len_batch,
                    ljm=ljm / len_batch,
                ),
            )

    def encode(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        N: int = 1,
        **kwargs,
    ) -> ModelOutput:
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        mcmc_steps = kwargs.pop("mcmc_steps", 100)
        n_lf = kwargs.pop("n_lf", 10)
        eps_lf = kwargs.pop("eps_lf", 0.01)
        return_mean = kwargs.pop("return_mean", False)
        if len(cond_mod) == self.n_modalities:
            output = self.joint_encoder(inputs.data)
            sample_shape = [] if N == 1 else [N]
            if return_mean:
                z = torch.stack([output.embedding] * N) if N > 1 else output.embedding
            else:
                z = dist.Normal(
                    output.embedding, torch.exp(0.5 * output.log_covariance)
                ).rsample(sample_shape)
            if N > 1 and kwargs.pop("flatten", False):
                N, l, d = z.shape
                z = z.reshape(l * N, d)
            return ModelOutput(z=z, one_latent_space=True)

        elif len(cond_mod) != 1:
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

        elif len(cond_mod) == 1:
            cond_mod = cond_mod[0]
            dcca_embed = self.DCCA_module.networks[cond_mod](
                inputs.data[cond_mod]
            ).embedding
            output = self.encoders[cond_mod](dcca_embed)
            sample_shape = [] if N == 1 else [N]

            z0 = dist.Normal(
                output.embedding, torch.exp(0.5 * output.log_covariance)
            ).rsample(sample_shape)
            flow_output = self.flows[cond_mod].inverse(
                z0.reshape(-1, self.latent_dim)
            )  # The reshaping is because MAF flows doesn't handle
            # any shape of input data (*,*input_dim)
            z = flow_output.out.reshape(z0.shape)

            if N > 1 and kwargs.pop("flatten", False):
                z = z.reshape(-1, self.latent_dim)

            return ModelOutput(z=z, one_latent_space=True)
        else:
            raise AttributeError(
                f"Modality of name {cond_mod} not handled. The"
                f" modalities that can be encoded are {list(self.encoders.keys())}"
            )

    def sample_from_moe_subset(self, subset: list, data: dict):
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
                encoder_output = self.encoders[m](
                    self.DCCA_module.networks[m](data[m][indices == m]).embedding
                )
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
            flow_output = self.flows[m](z)
            vae_output = self.encoders[m](
                self.DCCA_module.networks[m](data[m]).embedding
            )
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
            lnqzs += log_q_z0 + flow_output.log_abs_det_jac  # n_data_points x 1

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
            f"starting to sample from poe_subset, divide prior = {divide_prior}"
        )

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
        if ax is not None:
            ax.plot(pos[:, 0], pos[:, 1])
            ax.quiver(pos[:, 0], pos[:, 1], grad[:, 0], grad[:, 1])

            # plt.savefig('monitor_hmc.png')
        # 1/0
        print(acc_nbr[:10] / mcmc_steps)
        sh = (n_data, self.latent_dim) if K == 1 else (K, n_data, self.latent_dim)
        z = z.detach().resize(*sh)
        return z.detach()
