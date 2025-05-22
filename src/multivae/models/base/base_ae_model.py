import logging
from copy import deepcopy
from typing import Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder

from ...data.datasets.base import MultimodalBaseDataset
from ..nn.default_architectures import BaseDictDecoders, BaseDictEncoders
from .base_config import BaseMultiVAEConfig
from .base_model import BaseModel
from .base_utils import set_decoder_dist

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseMultiVAE(BaseModel):
    """Base class for Multimodal VAE models.

    Args:
        model_config (BaseMultiVAEConfig): An instance of BaseMultiVAEConfig in which any model's
            parameters is made available.

        encoders (Dict[str, ~pythae.models.nn.base_architectures.BaseEncoder]): A dictionary containing
            the modalities names and the encoders for each modality. Each encoder is an instance of
            Pythae's BaseEncoder. Default: None.

        decoders (Dict[str, ~pythae.models.nn.base_architectures.BaseDecoder]): A dictionary containing
            the modalities names and the decoders for each modality. Each decoder is an instance of
            Pythae's BaseDecoder.


    """

    def __init__(
        self,
        model_config: BaseMultiVAEConfig,
        encoders: dict = None,
        decoders: dict = None,
    ):
        super().__init__(model_config)

        # Set basic attributes
        self.model_name = "BaseMultiVAE"
        self.n_modalities = model_config.n_modalities
        self.input_dims = model_config.input_dims
        self.latent_dim = model_config.latent_dim
        self.device = None
        self.multiple_latent_spaces = False  # Default value, this field must be changed
        # in models using multiple latent spaces
        self.use_likelihood_rescaling = model_config.uses_likelihood_rescaling

        # Check the coherence between n_modalities and input_dims
        self.check_input_dims(model_config)

        # Set the encoders
        if encoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide encoders or input dims for the modalities in the model_config."
                )
            encoders = self.default_encoders(model_config)
        else:
            self.model_config.custom_architectures.append("encoders")

        # Set the decoders
        if decoders is None:
            if self.input_dims is None:
                raise AttributeError(
                    "Please provide decoders or input dims for the modalities in the model_config."
                )
            decoders = self.default_decoders(model_config)
        else:
            self.model_config.custom_architectures.append("decoders")

        # Check the coherence between encoders and decoders and model configuration
        self.sanity_check(encoders, decoders)
        self.set_decoders(decoders)
        self.set_encoders(encoders)
        self.modalities_name = list(self.decoders.keys())

        # Set the rescale factors
        self.rescale_factors = self.set_rescale_factors()

        # Set the output decoder distributions
        if model_config.decoders_dist is None:
            model_config.decoders_dist = {k: "normal" for k in self.encoders}
        if model_config.decoder_dist_params is None:
            model_config.decoder_dist_params = {}
        self.set_decoders_dist(
            model_config.decoders_dist, deepcopy(model_config.decoder_dist_params)
        )

    def set_decoders_dist(self, recon_dict, dist_params_dict):
        """Set the reconstruction losses functions decoders_dist
        and the log_probabilites functions recon_log_probs.
        recon_log_probs is the normalized negative version of recon_loss and is used only for
        likelihood estimation.
        """
        self.recon_log_probs = {}

        for k in recon_dict:
            self.recon_log_probs[k] = set_decoder_dist(
                recon_dict[k], dist_params_dict.get(k, {})
            )

        # TODO : add the possibility to provide custom reconstruction loss and in that case use the negative
        # reconstruction loss as the log probability.

    def check_input_dims(self, model_config):
        """Check that the input dimensions are coherent with the provided number of modalities."""
        if model_config.input_dims is not None:
            if len(model_config.input_dims.keys()) != model_config.n_modalities:
                raise AttributeError(
                    f"The provided number of input_dims {len(model_config.input_dims)} doesn't"
                    f"match the number of modalities ({model_config.n_modalities} in model config "
                )
        return

    def set_rescale_factors(self):
        """Set the rescale factors for the reconstruction losses.
        When using likelihood rescaling, the rescale factors are used to compute the
        reconstruction losses.
        """
        if self.use_likelihood_rescaling:
            # If rescale factors are provided, use them
            if self.model_config.rescale_factors is not None:
                rescale_factors = self.model_config.rescale_factors
            # If rescale factors are not provided, compute them from input dimensions
            elif self.input_dims is None:
                raise AttributeError(
                    " inputs_dim is None but (use_likelihood_rescaling = True"
                    " in model_config)"
                    " To compute default likelihood rescalings we need the input dimensions."
                    " Please provide a valid dictionary for input_dims or provide rescale_factors"
                    " in the model_config."
                )
            else:
                max_dim = max(*[np.prod(self.input_dims[k]) for k in self.input_dims])
                rescale_factors = {
                    k: max_dim / np.prod(self.input_dims[k]) for k in self.input_dims
                }
        else:
            rescale_factors = {k: 1 for k in self.encoders}
        return rescale_factors

    def sanity_check(self, encoders, decoders):
        """Check coherences between the encoders, decoders and model configuration."""
        if self.n_modalities != len(encoders.keys()):
            raise AttributeError(
                f"The provided number of encoders {len(encoders.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )

        if self.n_modalities != len(decoders.keys()):
            raise AttributeError(
                f"The provided number of decoders {len(decoders.keys())} doesn't"
                f"match the number of modalities ({self.n_modalities} in model config "
            )

        if encoders.keys() != decoders.keys():
            raise AttributeError(
                "The names of the modalities in the encoders dict doesn't match the names of the modalities"
                " in the decoders dict."
            )

        # If input_dims is provided, check that the modalities'names are coherent with encoders/decoders
        if self.input_dims is not None:
            if self.input_dims.keys() != encoders.keys():
                raise KeyError(
                    f"Warning! : The modalities names in model_config.input_dims : {list(self.input_dims.keys())}"
                    f" do not match the modalities names in encoders : {list(encoders.keys())}"
                )

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
            cond_mod (Union[list, str]): Either 'all' or a list of str containing the modalities names to condition on.
            N (int) : The number of encodings to sample for each datapoint. Default to 1.

        """
        # If the input cond_mod is a string : convert it to a list
        if isinstance(cond_mod, str):
            if cond_mod == "all":
                cond_mod = list(self.encoders.keys())
            elif cond_mod in self.encoders.keys():
                cond_mod = [cond_mod]
            else:
                raise AttributeError(
                    'If cond_mod is a string, it must either be "all" or a modality name'
                    f" The provided string {cond_mod} is neither."
                )

        ignore_incomplete = kwargs.pop("ignore_incomplete", False)
        # Deal with incomplete datasets
        if hasattr(inputs, "masks") and not ignore_incomplete:
            # Check that all modalities in cond_mod are available for all samples points.
            mods_avail = torch.tensor(True)
            for m in cond_mod:
                mods_avail = torch.logical_and(mods_avail, inputs.masks[m])
            if not torch.all(mods_avail):
                raise AttributeError(
                    "You tried to encode a incomplete dataset conditioning on",
                    f"modalities {cond_mod}, but some samples are not available"
                    "in all those modalities.",
                )
        return ModelOutput(cond_mod=cond_mod, z=None, one_latent_space=None)

    def decode(self, embedding: ModelOutput, modalities: Union[list, str] = "all"):
        """Decode a latent variable z in all modalities specified in modalities.

        Args:
            embedding (ModelOutput): contains the latent variables. It must have the same format as the
                output of the encode function.
            modalities (Union(List, str), Optional): the modalities to decode from z. Default to 'all'.

        Returns:
            ModelOutput : containing a tensor per modality name.
        """
        self.eval()
        with torch.no_grad():
            if modalities == "all":
                modalities = list(self.decoders.keys())
            elif isinstance(modalities, str):
                modalities = [modalities]

            try:
                if embedding.one_latent_space:
                    z = embedding.z
                    outputs = ModelOutput()
                    for m in modalities:
                        outputs[m] = self.decoders[m](z).reconstruction
                    return outputs
                else:
                    z_content = embedding.z
                    outputs = ModelOutput()
                    for m in modalities:
                        z = torch.cat([z_content, embedding.modalities_z[m]], dim=-1)
                        outputs[m] = self.decoders[m](z).reconstruction
                    return outputs
            except:
                raise ValueError(
                    "There was an error during decode. "
                    " Check that the format for the embedding is correct:"
                    "it must be a ModelOuput instance and "
                    "embedding.z must be a Tensor of shape (batch_size, *latent_shape)"
                    "If you used the encode function with N>1 to generate the embedding,"
                    " you need to pass flatten=True to have the right format for decoding."
                )

    def predict(
        self,
        inputs: MultimodalBaseDataset,
        cond_mod: Union[list, str] = "all",
        gen_mod: Union[list, str] = "all",
        N: int = 1,
        flatten: bool = False,
        **kwargs,
    ):
        """Generate in all modalities conditioning on a subset of modalities.

        Args:
            inputs (MultimodalBaseDataset): The data to condition on. It must contain at least the modalities
                contained in cond_mod.
            cond_mod (Union[list, str], optional): The modalities to condition on. Defaults to 'all'.
            gen_mod (Union[list, str], optional): The modalities to generate. Defaults to 'all'.
            N (int) : Number of samples to generate. Default to 1.
            flatten (int) : If N>1 and flatten is False, the returned samples have dimensions (N,len(inputs),...).
                Otherwise, the returned samples have dimensions (len(inputs)*N, ...)

        Returns:
            ~pythae.models.base.base_utils.ModelOutput

        ..codeblock :
            >>> predictions = model.predict(test_set, cond_mod = ['modality1', 'modality2'], gen_mod='modality3')
            >>> predictions.modality3


        """
        self.eval()
        ignore_incomplete = kwargs.pop("ignore_incomplete", False)
        z = self.encode(
            inputs,
            cond_mod,
            N=N,
            flatten=True,
            ignore_incomplete=ignore_incomplete,
            **kwargs,
        )
        output = self.decode(z, gen_mod)
        n_data = len(z.z) // N
        if not flatten and N > 1:
            for m in output.keys():
                output[m] = output[m].reshape(N, n_data, *output[m].shape[1:])
        return output

    def forward(self, inputs: MultimodalBaseDataset, **kwargs) -> ModelOutput:
        """Main forward pass outputing the VAE outputs
        This function should output a :class:`~pythae.models.base.base_utils.ModelOutput` instance
        gathering all the model outputs.

        Args:
            inputs (BaseDataset): The training data with labels, masks etc...

        Returns:
            ModelOutput: A ModelOutput instance providing the outputs of the model.

        .. note::
            The loss must be computed in this forward pass and accessed through
            ``loss = model_output.loss``
        """
        raise NotImplementedError()

    def update(self):
        """Method that allows model update during the training (at the end of a training epoch).

        If needed, this method must be implemented in a child class.

        By default, it does nothing.
        """
        pass

    def default_encoders(self, model_config) -> nn.ModuleDict:
        return BaseDictEncoders(self.input_dims, model_config.latent_dim)

    def default_decoders(self, model_config) -> nn.ModuleDict:
        return BaseDictDecoders(self.input_dims, model_config.latent_dim)

    def set_encoders(self, encoders: dict) -> None:
        """Set the encoders of the model."""
        self.encoders = nn.ModuleDict()
        for modality in encoders:
            encoder = encoders[modality]
            if not issubclass(type(encoder), BaseEncoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, encoder must inherit from BaseEncoder class from "
                        "pythae.models.base_architectures.BaseEncoder. Refer to documentation."
                    )
                )

            self.encoders[modality] = encoder

    def set_decoders(self, decoders: dict) -> None:
        """Set the decoders of the model."""
        self.decoders = nn.ModuleDict()
        for modality in decoders:
            decoder = decoders[modality]
            if not issubclass(type(decoder), BaseDecoder):
                raise AttributeError(
                    (
                        f"For modality {modality}, decoder must inherit from BaseDecoder class from "
                        "pythae.models.base_architectures.BaseDecoder. Refer to documentation."
                    )
                )
            self.decoders[modality] = decoder

    def compute_joint_nll(
        self, inputs: MultimodalBaseDataset, K: int = 1000, batch_size_K: int = 100
    ):
        raise NotImplementedError

    def generate_from_prior(self, n_samples, **kwargs):
        """Generate latent samples from the prior distribution.
        This is the base class in which we consider a static standard Normal Prior.
        This may be overwritten in subclasses.

        Args:
            n_samples (int): number of samples to generate
            **kwargs: additional arguments
        Returns:
            ModelOutput: A ModelOutput instance containing the generated samples
        """
        sample_shape = (
            [n_samples, self.latent_dim] if n_samples > 1 else [self.latent_dim]
        )
        z = dist.Normal(0, 1).rsample(sample_shape).to(self.device)
        return ModelOutput(z=z, one_latent_space=True)

    def compute_cond_nll(
        self,
        inputs: MultimodalBaseDataset,
        subset: Union[list, tuple],
        pred_mods: Union[list, tuple],
        k_iwae=1000,
    ):
        r"""Compute the conditional likelihood :math: `ln p(x_{pred}|x_{cond})`` with MonteCarlo Sampling and the approximation :
        .. math::
                \ln p(x_{pred)|x_{cond}) = \frac{1}{K}\sum_{z^{(i)} ~ q(z^{(i)}|x_{cond}), i=1}^{K} \ln p(x_{pred}|z^{(i)}).

        Args:
            inputs (MultimodalBaseDataset): the data to compute the likelihood on.
            cond_mod (str): the modality to condition on
            gen_mod (str): the modality to condition on
            K (int, optional): number of samples per batch. Defaults to 1000.

        Returns:
            dict: Contains the negative log-likelihood for each modality in pred_mods.
        """
        cnll = {m: [] for m in pred_mods}

        for _ in range(k_iwae):
            # Encode the inputs conditioning on subset
            encode_output = self.encode(inputs, subset)
            # Decode
            decode_output = self.decode(encode_output, pred_mods)
            # Compute ln(p(x_{pred}|z)) for each modality
            for mod in pred_mods:
                recon = decode_output[mod]  # (n_data, *recon_size )
                lpxz = (
                    self.recon_log_probs[mod](recon, inputs.data[mod])
                    .reshape(recon.size(0), -1)
                    .sum(-1)
                )
                cnll[mod].append(lpxz)  # (n_data)

        for mod, c in cnll.items():
            cnll[mod] = torch.stack(c)  # stack the results of mini_batches of K samples
            cnll[mod] = torch.logsumexp(cnll[mod], dim=0) - np.log(
                k_iwae
            )  # average over the samples
            cnll[mod] = -torch.sum(cnll[mod]) / len(
                cnll[mod]
            )  # average over the data points and take negative

        return cnll
