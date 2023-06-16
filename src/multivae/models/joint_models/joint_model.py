import logging
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
from pythae.models.nn.base_architectures import BaseEncoder

from ...data import MultimodalBaseDataset
from ..base import BaseMultiVAE
from ..nn.default_architectures import MultipleHeadJointEncoder
from .joint_model_config import BaseJointModelConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseJointModel(BaseMultiVAE):
    """
    Base Class for models using a joint encoder.

    Args:

        model_config (BaseJointModelConfig): The configuration of the model.

        encoders (Dict[BaseEncoder]): A dictionary containing the modalities names and the encoders for each
            modality. Each encoder is an instance of Pythae's BaseEncoder class.

        decoder (Dict[BaseDecoder]): A dictionary containing the modalities names and the decoders for each
            modality. Each decoder is an instance of Pythae's BaseDecoder class.

        joint_encoder (BaseEncoder) : An instance of BaseEncoder that takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.
    """

    def __init__(
        self,
        model_config: BaseJointModelConfig,
        encoders: dict = None,
        decoders: dict = None,
        joint_encoder: Union[BaseEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders)

        if joint_encoder is None:
            # Create a MultiHead Joint Encoder MLP
            joint_encoder = MultipleHeadJointEncoder(self.encoders, model_config)
        else:
            self.model_config.custom_architectures.append("joint_encoder")

        self.set_joint_encoder(joint_encoder)

    def set_joint_encoder(self, joint_encoder):
        "Checks that the provided joint encoder is an instance of BaseEncoder."

        if not issubclass(type(joint_encoder), BaseEncoder):
            raise AttributeError(
                (
                    f"The joint encoder must inherit from BaseEncoder class from "
                    "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                )
            )
        self.joint_encoder = joint_encoder

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """
        Return the estimated negative log-likelihood summed over the input batch.
        The negative log-likelihood is estimated using importance sampling.

        Args:
            inputs : the data to compute the joint likelihood"""

        # First compute all the parameters of the joint posterior q(z|x,y)

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
                for mod in inputs.data:
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, *decoder_output_shape)
                    x_m = inputs.data[mod][i]  # (*input_shape)
                    lpx_zs += (
                        self.recon_log_probs[mod](recon, x_m)
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

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
