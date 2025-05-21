import logging
from typing import Union

import numpy as np
import torch
import torch.distributions as dist

from multivae.models.nn import BaseJointEncoder

from ...data import MultimodalBaseDataset
from ..base import BaseMultiVAE
from ..nn.default_architectures import MultipleHeadJointEncoder
from .joint_model_config import BaseJointModelConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseJointModel(BaseMultiVAE):
    """Base Class for models using a joint encoder.

    Args:
        model_config (BaseJointModelConfig): The configuration of the model.

        encoders (Dict[BaseEncoder]): A dictionary containing the modalities names and the encoders for each
            modality. Each encoder is an instance of Pythae's BaseEncoder class.

        decoder (Dict[BaseDecoder]): A dictionary containing the modalities names and the decoders for each
            modality. Each decoder is an instance of Pythae's BaseDecoder class.

        joint_encoder (BaseJointEncoder) : Takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.
    """

    def __init__(
        self,
        model_config: BaseJointModelConfig,
        encoders: dict = None,
        decoders: dict = None,
        joint_encoder: Union[BaseJointEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders)

        if joint_encoder is None:
            # Create a MultiHead Joint Encoder MLP
            joint_encoder = self.default_joint_encoder(model_config)
        else:
            self.model_config.custom_architectures.append("joint_encoder")

        self.set_joint_encoder(joint_encoder)

    def default_joint_encoder(self, model_config):
        return MultipleHeadJointEncoder(self.encoders, model_config)

    def set_joint_encoder(self, joint_encoder):
        """Checks that the provided joint encoder is an instance of BaseJointEncoder."""
        if not issubclass(type(joint_encoder), BaseJointEncoder):
            raise AttributeError(
                (
                    "The joint encoder must inherit from "
                    "~multivae.models.nn.default_architectures.BaseJointEncoder . Refer to documentation."
                )
            )
        self.joint_encoder = joint_encoder

    def forward(self, inputs, **kwargs):
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The inputs have masks but this model is not compatible with incomplete dataset."
            )

    def encode(self, inputs, cond_mod="all", N=1, return_mean=False, **kwargs):
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The inputs have masks but this model is not compatible with incomplete dataset."
            )
        return super().encode(inputs, cond_mod, N, **kwargs)

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        """Estimate the negative joint likelihood.

        Args:
            inputs (MultimodalBaseDataset) : a batch of samples.
            K (int) : the number of importance samples for the estimation. Default to 1000.
            batch_size_K (int) : Default to 100.

        Returns:
            The negative log-likelihood summed over the batch.
        """
        # Check that the dataset is not incomplete.
        self.eval()
        if hasattr(inputs, "masks"):
            raise AttributeError(
                "The inputs contains masks but this model is not compatible with incomplete dataset."
            )

        # Compute the parameters of the joint posterior
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance
        sigma = torch.exp(0.5 * log_var)

        # Sample K latents from the joint posterior
        qz_xy = dist.Normal(mu, sigma)
        z_joint = qz_xy.rsample([K]).permute(1, 0, 2)  # n_data x K x latent_dim
        n_data, _, _ = z_joint.shape

        # Then iter on each datapoint to compute the iwae estimate of ln(p(x))
        ll = 0
        for i in range(n_data):
            start_idx = 0
            stop_idx = min(start_idx + batch_size_K, K)
            lnpxs = []
            while start_idx < stop_idx:
                latents = z_joint[i][start_idx:stop_idx]

                # Compute ln p(x_m|z) for z in latents and for each modality m
                lpx_zs = 0
                for mod in inputs.data:
                    decoder = self.decoders[mod]
                    recon = decoder(latents)[
                        "reconstruction"
                    ]  # (batch_size_K, *decoder_output_shape)
                    x_m = inputs.data[mod][i]  # (*input_shape)
                    lpx_zs += (
                        self.recon_log_probs[mod](
                            recon, torch.stack([x_m] * len(recon))
                        )
                        .reshape(recon.size(0), -1)
                        .sum(-1)
                    )

                # Compute ln(p(z))
                prior = dist.Normal(0, 1)
                lpz = prior.log_prob(latents).sum(dim=-1)

                # Compute posteriors -ln(q(z|X))
                qz_xy = dist.Normal(mu[i], sigma[i])
                lqz_xy = qz_xy.log_prob(latents).sum(dim=-1)

                ln_px = torch.logsumexp(lpx_zs + lpz - lqz_xy, dim=0)
                lnpxs.append(ln_px)

                # next batch
                start_idx += batch_size_K
                stop_idx = min(stop_idx + batch_size_K, K)

            ll += torch.logsumexp(torch.Tensor(lnpxs), dim=0) - np.log(K)

        return -ll
