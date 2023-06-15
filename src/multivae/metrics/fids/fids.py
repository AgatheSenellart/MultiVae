from itertools import combinations

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput
from scipy import linalg
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Resize

from multivae.data import MultimodalBaseDataset
from multivae.data.utils import set_inputs_to_device
from multivae.models.base import BaseMultiVAE
from multivae.samplers import BaseSampler

from ..base.evaluator_class import Evaluator
from .fids_config import FIDEvaluatorConfig
from .inception_networks import wrapper_inception

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x


class adapt_shape_for_fid(torch.nn.Module):
    """
    Transform an input so that each sample has three dimensions with three channels.
    (batch_size, 2,h,w). The input is assumed to be batched.
    """

    def __init__(self, resize=True, **kwargs) -> None:
        super().__init__(**kwargs)
        if resize:
            self.resize = Resize((299, 299))
        else:
            self.resize = None

    def forward(self, x):
        if len(x.shape) == 1:  # (n_data,)
            x = x.unsqueeze(1)
        if len(x.shape) == 2:  # (n_data, n)
            x = x.unsqueeze(1)
        if len(x.shape) == 3:  # (n_data, n, m)
            x = x.unsqueeze(1)
        if len(x.shape) == 4:
            if x.shape[1] == 1:
                # Add channels to have 3 channels
                x = torch.cat([x for _ in range(3)], dim=1)
            elif x.shape[1] == 2:
                n, ch, h, w = x.shape
                x = torch.cat([x, torch.zeros(n, 1, h, w)], dim=1)
            else:
                x = x[:, :3, :, :]

            if self.resize is not None:
                return self.resize(x)
            else:
                return x
        else:
            raise AttributeError("Can't visualize data with more than 3 dimensions")


class FIDEvaluator(Evaluator):
    """
    Class for computing likelihood metrics.

    Args:
        model (BaseMultiVAE) : The model to evaluate.
        classifiers (dict) : A dictionary containing the pretrained classifiers to use for the coherence evaluation.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        sampler (Basesampler) : The sampler used to generate from the latent space.
            If None is provided, the latent codes are generated from prior. Default to None.
        custom_encoder (torch.nn.Module) : If you desire, you can provide our own embedding architecture to use
            instead of the InceptionV3 model to compute Fréchet Distances.
            By default, the pretrained InceptionV3 network is used. Default to None.
        transform (torchvision.Transforms) : To apply to the images before computing the embeddings. If None is provided
            a default resizing to (3,299,299) is applied. Default to None.
    """

    def __init__(
        self,
        model: BaseMultiVAE,
        test_dataset,
        output=None,
        eval_config=FIDEvaluatorConfig(),
        sampler: BaseSampler = None,
        custom_encoder=None,
        transform=None,
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config, sampler)

        if custom_encoder is not None:
            self.model_fd = custom_encoder
        else:
            self.model_fd = wrapper_inception(
                dims=eval_config.dims_inception,
                device=self.device,
                path_state_dict=eval_config.inception_weights_path,
            )
        if transform is not None:
            self.inception_transform = transform
        else:
            self.inception_transform = adapt_shape_for_fid()

    def get_frechet_distance(self, mod, generate_latent_function):
        """
        Calculates the activations of the pool_3 layer for all images.
        """
        self.model.eval()
        activations = [[], []]

        for batch in tqdm(self.test_loader):
            batch = set_inputs_to_device(batch, self.device)
            # Compute activations for true data
            data = self.inception_transform(batch.data[mod]).to(self.device)
            pred = self.model_fd(data)
            del data
            activations[0].append(pred)

            # Compute activations for generated data
            latents = generate_latent_function(len(pred), inputs=batch)
            latents.z = latents.z.to(self.device)

            samples = self.model.decode(latents, modalities=mod)
            data_gen = self.inception_transform(samples[mod])
            del samples
            pred_gen = self.model_fd(data_gen)
            activations[1].append(pred_gen)
            del data_gen

        activations = [np.concatenate(l, axis=0) for l in activations]

        # Compute activation statistics
        mus = [np.mean(act, axis=0) for act in activations]
        sigmas = [np.cov(act, rowvar=False) for act in activations]
        fd = self.calculate_frechet_distance(mus[0], sigmas[0], mus[1], sigmas[1])
        return fd

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        r"""Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians :math:`X_1 \sim
        \mathcal{N}(\mu_1, C_1)`
        and :math:`X_2 \sim \mathcal{N}(\mu_2, C_2)` is
        :math:`d^2 = \lVert \mu_1 - \mu_2\rVert^2 + \mathrm{Tr}(C_1 + C_2 -
        2\sqrt{(C_1\cdot C_2)})`.
        Stable version by Dougal J. Sutherland.

        Args:
            mu1 (numpy.ndarray): Numpy array containing the activations of a layer of the
                    inception net (like returned by the function 'get_predictions')
                    for generated samples.
            mu2 (numpy.ndarray): The sample mean over activations, precalculated on an
                    representative data set.
            sigma1 (numpy.ndarray): The covariance matrix over activations for generated samples.
            sigma2 (numpy.ndarray): The covariance matrix over activations, precalculated on an
                    representative data set.

        Return:
            numpy.ndarray : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, (
            f"Training and test mean vectors have different lengths. mu1 has shape {mu1.shape}"
            f"whereas mu2 has shape {mu2.shape}"
        )
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            self.logger.info(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def unconditional_fids(self):
        """
        Generate data from prior or sampler fitted in the latent space
        and compute the FID for each modality.

        Returns:
            ~pythae.models.base.base_utils.ModelOutput: FIDs for all modalities.
        """
        output = dict()
        if self.sampler is None:
            generate_function = self.model.generate_from_prior
        else:
            generate_function = self.sampler.sample

        sampler_name = "prior" if self.sampler is None else self.sampler.name
        for mod in self.model.encoders:
            self.logger.info(f"Start computing FID for modality {mod}")
            fd = self.get_frechet_distance(mod, generate_function)
            output[f"fd_{mod}_sampler_{sampler_name}"] = fd
            self.logger.info(
                f"The FD for modality {mod} with sampler {sampler_name} is {fd}"
            )
        self.metrics.update(output)

        return ModelOutput(**output)

    def eval(self):
        self.unconditional_fids()
        self.log_to_wandb()

        return ModelOutput(**self.metrics)

    def compute_fid_from_conditional_generation(self, subset, gen_mod):
        """
        Generate samples from the conditional distribution conditioned on subset and compute
        Frechet distance for gen_mod.
        """

        generate_function = lambda n_samples, inputs: self.model.encode(
            inputs=inputs, cond_mod=subset
        )
        fd = self.get_frechet_distance(gen_mod, generate_function)
        self.logger.info(
            f"The FD for modality {gen_mod} computed from subset={subset} is {fd}"
        )

        self.metrics[f"Conditional FD from {subset} to {gen_mod}"] = fd
        return fd

    def compute_all_conditional_fids(self, gen_mod):
        """
        For all subsets in modalities \gen mod, compute the FID when generating
        images from the subsets.
        """

        modalities = [k for k in self.model.encoders if k != gen_mod]
        fds = []
        for n in range(1, len(modalities) + 1):
            s = modalities[:n]
            fd = self.compute_fid_from_conditional_generation(s, gen_mod)
            fds.append(fd)

        self.log_to_wandb()

        return ModelOutput(**self.metrics)
