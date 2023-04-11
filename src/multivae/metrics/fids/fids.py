from itertools import combinations

import numpy as np
import torch
from pythae.models.base.base_utils import ModelOutput
from torch.utils.data import DataLoader

from multivae.data import MultimodalBaseDataset
from multivae.models.base import BaseMultiVAE

from ..base.evaluator_class import Evaluator
from .fids_config import FIDEvaluatorConfig
from .inception_networks import wrapper_inception
from torchvision.transforms import Resize
from torch.utils.data import TensorDataset

try :
    from tqdm import tqdm
except:
    def tqdm(x):
        return x

class FIDEvaluator(Evaluator):
    """
    Class for computing likelihood metrics.

    Args:
        model (BaseMultiVAE) : The model to evaluate.
        classifiers (dict) : A dictionary containing the pretrained classifiers to use for the coherence evaluation.
        test_dataset (MultimodalBaseDataset) : The dataset to use for computing the metrics.
        output (str) : The folder path to save metrics. The metrics will be saved in a metrics.txt file.
        eval_config (EvaluatorConfig) : The configuration class to specify parameters for the evaluation.
        custom_encoder (torch.nn.Module) : If you desire, you can provide our own embedding architecture to use
            instead of the InceptionV3 model to compute Fréchet Distances. 
            By default, the pretrained InceptionV3 network is used. Default to None.
        transform (torchvision.Transforms) : To apply to the images before computing the embeddings. Default to
            Resize((299, 299)). 
    """

    def __init__(
        self, model : BaseMultiVAE, test_dataset, output=None, eval_config=FIDEvaluatorConfig(),
        custom_encoder = None,
        transform = Resize((299, 299)),
    ) -> None:
        super().__init__(model, test_dataset, output, eval_config)
        
        if custom_encoder is not None:
            self.model_fd = custom_encoder
        else :
            self.model_fd = wrapper_inception(dims=eval_config.dims_inception,
                                           device=self.device,
                                           path_state_dict=eval_config.inception_weights_path)
        

        self.inception_transform = transform
    
    
    def get_frechet_distance(self, mod,generate_latent_function):
        """
        Calculates the activations of the pool_3 layer for all images.
        """
        self.model.eval()
        activations = [[],[]]
        
        for batch in tqdm(self.test_loader):
            # Compute activations for true data
            data = self.inception_transform(batch.data[mod]).to(self.device)
            pred = self.model_fd(data)
            activations[0].append(pred)
            
            # Compute activations for generated data
            latents = generate_latent_function(n_samples= len(data))
            latents.z = latents.z.to(self.device)
            samples = self.model.decode(latents,modalities=mod)
            data_gen = self.inception_transform(samples[mod])
            pred_gen = self.model_fd(data_gen)
            print(pred_gen.shape)
            
            activations[1].append(pred_gen)

        activations = [np.concatenate(l, axis=0) for l in activations]

        # Compute activation statistics
        mus = [np.mean(act, axis=0) for act in activations]
        sigmas = [np.cov(act, rowvar=False) for act in activations]
        print(mus[0].shape, mus[1].shape)
        fd = self.calculate_frechet_distance(mus[0], sigmas[0],mus[1], sigmas[1])
        return fd

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            f'Training and test mean vectors have different lengths. mu1 has shape {mu1.shape}'\
            f'whereas mu2 has shape {mu2.shape}'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = np.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
        
        
    def eval(self):
        
        output = ModelOutput()
        
        # Generate data from the prior and computes FID for each modality
        generate_function = self.model.generate_from_prior
        for mod in self.model.encoders:
            self.logger.info(f"Start computing FID for modality {mod}")
            fd = self.get_frechet_distance(mod,generate_function)
            output[f'fd_{mod}'] = fd
            self.logger.info(f'The FD for modality {mod} is {fd}')
        
        # TODO : Comput Frechet distances for conditional generation
        
        return output
        
                
                
                

            
