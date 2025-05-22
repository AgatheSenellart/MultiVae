r"""Implementation of the DMVAE model from
"Private-Shared Disentangled Multimodal VAE for Learning of Latent
Representations" (Lee & Pavlovic 2021)(https://par.nsf.gov/servlets/purl/10297662).

This model is an aggregated model with a shared latent variable :math:`z_s` and modality-specific latent variables :math:`z_{p_i}`.
The joint posterior is a Product-Of-Experts:

.. math::
    q(z_s|X) \propto p(z_s)\prod_{i=1}^{M} q(z_s|x_i)

The joint ELBO writes:

.. math::
    &\sum_i \lambda_i \mathbb{E}_{\substack{q_{\phi}(z_{p_i}|x_i)} \\ q_{\phi}(z_s|X)}\left[ \log p_{\theta}(x_i|z_{p_i},z_s)\right] \\
    & -KL(q_{\phi}(z_{p_i}|x_i)||p(z_{p_i})) - KL(q_{\phi}(z_s|X)||p(z_s))\\
    & + \sum_j \lambda_i \mathbb{E}_{\substack{q_{\phi}(z_{p_i}|x_i)} \\ q_{\phi}(z_s|x_j)}\left[ \log p_{\theta}(x_i|z_{p_i},z_s)\right] \\
    & -KL(q_{\phi}(z_{p_i}|x_i)||p(z_{p_i})) - KL(q_{\phi}(z_s|x_j)||p(z_s))

This loss incorporates differents ELBOS, using either the joint posterior or each of the unimodal posteriors.

.. note::
    This model can be used in the partially observed setting. In that case,  for each sample :math:`X`, we take the loss and the product-of-experts on 
    available modalities only.

"""

from .dmvae_config import DMVAEConfig
from .dmvae_model import DMVAE

__all__ = ["DMVAE", "DMVAEConfig"]
