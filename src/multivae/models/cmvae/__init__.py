r"""Implementation of "Deep Generative Clustering with Multimodal Diffusion Variational Autoencoders"
(Palumbo et al, 2023)(https://openreview.net/forum?id=k5THrhXDV3).

This model builds on the MMVAE+ by adding a Mixture-of-Gaussians prior on the shared latent space.
The generative model is as follows:

.. math::
    \begin{split}
    & c \sim \pi \\
    & z|c \sim \mathcal{N}(\mu_c, \Sigma_c) \\
    & \forall m,  w_m \sim p(w_m) \\
    & \forall m, x_m|z, w_m \sim p_{\theta}(x_m|z,w_m)\\
   
    \end{split}

The joint posterior for the latent space variable :math:`z` is a mixture-of-experts:

.. math::
    q_{\Phi_z}(z|X) = \frac{1}{M} \sum_m q_{\Phi_z}(z|x_m)

This model uses trainable auxiliary priors :math:`r_m(w_m)` for the modality specific latent spaces during training. 

The ELBO of the CMVAE model writes:

.. math::
   \frac{1}{M}\sum_{m=1}^{M} \mathbb E_{
    \substack{
        q_{\Phi_{z_m}}(z|x_m) \\ q_{\phi_{w_m}}(w_m|x_m) \\ q(c|z,X)
    }
    } \left[ G_{\Phi_{z},\phi_{w_m},\theta, \pi}(X,c,z,w_m) \right]

where

.. math::

    G_{\Phi_{z},\phi_{w_m},\theta, \pi}(X,c,z,w_m) &= \log p_{\theta}(x_m|z, w_m) + \sum_{n \neq m} \mathbb{E}_{\tilde{w_n} \sim r_n(w_n)}\left[ \log p_{\theta_n}(x_n|z, \tilde{w_n}) \right] \\
    
    & + \beta \log \left( \frac{p_{\pi}(c)p_{\theta}(z|c)p(w_m)}{q_{\Phi_z}(z|X)q_{\phi_m}(w_m|x_m)q(c|X,z)}\right)\\

In practice the ELBO is approximated using importance sampling with K> 1 samples. 
This method can also be used for clustering with an ad-hoc procedure for selecting the number of clusters a posteriori.

.. note::
    This model can be used in the partially observed setting. In that scenario, we adapt the model in a similar fashion as for the MMVAE+.

.. note::
    The diffusion decoders are not yet supported. 

"""

from .cmvae_config import CMVAEConfig
from .cmvae_model import CMVAE

__all__ = ["CMVAE", "CMVAEConfig"]
