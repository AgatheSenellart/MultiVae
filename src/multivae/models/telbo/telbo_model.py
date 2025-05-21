from typing import Dict, Union

import torch
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from ..base.base_utils import rsample_from_gaussian
from ..joint_models import BaseJointModel
from ..nn.base_architectures import BaseJointEncoder
from .telbo_config import TELBOConfig


class TELBO(BaseJointModel):
    """The Triple ELBO VAE model.

    Args:
        model_config (TELBOConfig): An instance of TELBOConfig in which any model's parameters is
            made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.

        joint_encoder (~multivae.models.nn.base_architectures.BaseJointEncoder) : takes all the modalities as an input.
            If none is provided, one is created from the unimodal encoders. Default : None.


    """

    def __init__(
        self,
        model_config: TELBOConfig,
        encoders: Dict[str, BaseEncoder] = None,
        decoders: Dict[str, BaseDecoder] = None,
        joint_encoder: Union[BaseJointEncoder, None] = None,
        **kwargs,
    ):
        super().__init__(model_config, encoders, decoders, joint_encoder, **kwargs)

        self.model_name = "TELBO"
        self.warmup = model_config.warmup
        self.reset_optimizer_epochs = [self.warmup]

        if model_config.lambda_factors is None:
            self.lambda_factors = self.rescale_factors
        else:
            self.lambda_factors = model_config.lambda_factors
        if model_config.gamma_factors is None:
            self.gamma_factors = self.rescale_factors
        else:
            self.gamma_factors = model_config.gamma_factors

    def _set_torch_no_grad_on_joint_vae(self):
        """Function used to freeze the parameters of the joint
        encoder and decoders after the warmup.
        """
        self.joint_encoder.requires_grad_(False)
        self.decoders.requires_grad_(False)

    def forward(self, inputs: MultimodalBaseDataset, **kwargs):
        """Forward pass of the model."""
        # Check that the dataset is not incomplete
        super().forward(inputs)

        epoch = kwargs.pop("epoch", 1)

        # First compute the joint ELBO
        joint_output = self.joint_encoder(inputs.data)
        mu, log_var = joint_output.embedding, joint_output.log_covariance

        z_joint = rsample_from_gaussian(mu, log_var)

        recon_loss = 0

        # Decode in each modality
        len_batch = 0
        for mod in self.decoders:
            x_mod = inputs.data[mod]
            len_batch = len(x_mod)
            recon_mod = self.decoders[mod](z_joint).reconstruction
            recon_loss += (
                -self.recon_log_probs[mod](recon_mod, x_mod) * self.lambda_factors[mod]
            ).sum()

        # Compute the KLD to the prior
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        if epoch <= self.warmup:
            return ModelOutput(
                recon_loss=recon_loss / len_batch,
                KLD=KLD / len_batch,
                loss=(recon_loss + KLD) / len_batch,
                metrics=dict(kld_joint=KLD, recon_joint=recon_loss / len_batch),
            )

        else:
            self._set_torch_no_grad_on_joint_vae()
            mod_elbos = {}
            loss = 0

            for mod in self.encoders:
                mod_output = self.encoders[mod](inputs.data[mod])
                mod_mu, mod_log_var = mod_output.embedding, mod_output.log_covariance

                mod_z = rsample_from_gaussian(mod_mu, mod_log_var)

                mod_recon = self.decoders[mod](mod_z).reconstruction
                mod_recon_loss = (
                    -self.recon_log_probs[mod](mod_recon, inputs.data[mod])
                    * self.gamma_factors[mod]
                )

                mod_kld = -0.5 * torch.sum(
                    1 + log_var - mod_mu.pow(2) - mod_log_var.exp()
                )
                mod_elbos[mod] = mod_recon_loss.sum() + mod_kld
                loss += mod_recon_loss.sum() + mod_kld

            return ModelOutput(loss=loss / len_batch, loss_sum=loss, metrics=mod_elbos)

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


        Returns:
            ModelOutput instance with fields:
                z (torch.Tensor (N,n_data, latent_dim))
                one_latent_space (bool) = True

        """
        self.eval()
        # Transform to list and check that dataset is complete
        cond_mod = super().encode(inputs, cond_mod, N, **kwargs).cond_mod

        # If one conditioning modality, use the modality encoder
        if len(cond_mod) == 1:
            cond_mod = cond_mod[0]
            output = self.encoders[cond_mod](inputs.data[cond_mod])
        # If all conditioning modalities, use the joint encoder
        elif len(cond_mod) == self.n_modalities:
            output = self.joint_encoder(inputs.data)

        else:
            raise ValueError(
                f" Conditioning on subset {cond_mod} is not handled. "
                f" Possible subsets are  {list(self.encoders.keys())} and 'all'. "
            )

        # Return mean or sample
        flatten = kwargs.pop("flatten", False)
        z = rsample_from_gaussian(
            output.embedding, output.log_covariance, N, return_mean, flatten
        )

        return ModelOutput(z=z, one_latent_space=True)
