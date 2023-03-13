from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.normalizing_flows.base import BaseNF, BaseNFConfig
from pythae.models.normalizing_flows.maf import MAF, MAFConfig
from torch.nn import ModuleDict

from multivae.models.nn.default_architectures import BaseDictEncoders, MultipleHeadJointEncoder

from ...data.datasets.base import MultimodalBaseDataset
from ..joint_models import BaseJointModel
from .jnf_dcca_config import JNFDccaConfig
from ..dcca import DCCA, DCCAConfig


class JNFDcca(BaseJointModel):

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
        dcca_networks : Dict[str, BaseEncoder] = None,
        **kwargs,
    ):
        
        
        self.dcca_config = DCCAConfig(n_modalities=model_config.n_modalities,
                                      embedding_dim=model_config.embedding_dcca_dim,
                                      use_all_singular_values=model_config.use_all_singular_values)
        
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
                dcca_networks = BaseDictEncoders(model_config.input_dims, model_config.embedding_dcca_dim)
        else:
            model_config.use_default_dcca_network = False
            
        
        # The default encoders for this model have (embedding_dcca_dim, ) as input_size
        if encoders is None:
            encoders_input_dims = {k: (model_config.embedding_dcca_dim,) for k in dcca_networks}
            encoders = BaseDictEncoders(encoders_input_dims,model_config.latent_dim)
        else :
            model_config.uses_default_encoders = False
            
        # The default joint_encoder for this model is engineered from the DCCA networks and
        # not from the encoders
        if joint_encoder is None:
            # Create a MultiHead Joint Encoder MLP
            joint_encoder = MultipleHeadJointEncoder(dcca_networks, model_config)
        else:
            model_config.use_default_joint = False
            
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)
        self.DCCA_module = DCCA(self.dcca_config,dcca_networks)


        if flows is None:
            flows = dict()
            for modality in self.encoders:
                flows[modality] = MAF(MAFConfig(input_dim=(model_config.latent_dim,)))
        else:
            self.model_config.use_default_flow = False

        self.set_flows(flows)
        self.model_name = "JNFDcca"
        self.warmup = model_config.warmup
        self.nb_epochs_dcca = model_config.nb_epochs_dcca
        self.reset_optimizer_epochs = [self.nb_epochs_dcca,
                                       self.nb_epochs_dcca+self.warmup]

        
        
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
                else :
                    raise AttributeError("The provided flows don't have the right input dim.")
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
                self.recon_losses[mod](recon_mod, x_mod) * self.rescale_factors[mod]
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
            dcca_embed = self.DCCA_module.networks[cond_mod](inputs.data[cond_mod]).embedding
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
