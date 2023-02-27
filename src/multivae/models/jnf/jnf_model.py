from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.normalizing_flows.base import BaseNF, BaseNFConfig
from pythae.models.normalizing_flows.maf import MAF, MAFConfig
from torch.nn import ModuleDict

from ...data.datasets.base import MultimodalBaseDataset
from ..joint_models import BaseJointModel
from .jnf_config import JNFConfig


class JNF(BaseJointModel):

    """
    The JNF model.

    Args:

        model_config (JNFConfig): An instance of JNFConfig in which any model's parameters is
            made available.

        encoders (Dict[str,BaseEncoder]): A dictionary containing the modalities names and the encoders for each
            modality. Each encoder is an instance of Pythae's BaseEncoder.

        decoders (Dict[str,BaseDecoder]): A dictionary containing the modalities names and the decoders for each
            modality. Each decoder is an instance of Pythae's BaseDecoder.

        joint_encoder (BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.

        flows (Dict[str,BaseNF]) : A dictionary containing the modalities names and the flows to use for
            each modality. If None is provided, a default MAF flow is used for each modality.


    """

    def __init__(
        self,
        model_config: JNFConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        flows: Dict[str, BaseNF] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        if flows is None:
            flows = dict()
            self.model_config.use_default_flow = True
            for modality in self.encoders:
                flows[modality] = MAF(MAFConfig(input_dim=(model_config.latent_dim,)))

        self.set_flows(flows)

        self.model_name = "JNF"
        self.warmup = model_config.warmup

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
                -0.5 * torch.sum((x_mod - recon_mod) ** 2)
            ) * self.rescale_factors[mod]

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if epoch < self.warmup:
            return ModelOutput(
                recon_loss=recon_loss / len_batch,
                KLD=KLD / len_batch,
                loss=-(recon_loss - KLD) / len_batch,
                metrics=dict(kld_prior=KLD, recon_loss=recon_loss / len_batch, ljm=0),
            )

        else:
            self._set_torch_no_grad_on_joint_vae()
            ljm = 0
            for mod in self.encoders:
                mod_output = self.encoders[mod](inputs.data[mod])
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
        self.eval()

        if type(cond_mod) == list and len(cond_mod) == 1:
            cond_mod = cond_mod[0]

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
            raise AttributeError(
                "Conditioning on a subset containing more than one modality "
                "is not yet implemented."
            )

        if cond_mod in self.input_dims.keys():
            output = self.encoders[cond_mod](inputs.data[cond_mod])
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
