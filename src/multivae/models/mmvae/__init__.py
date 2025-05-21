r"""Implementation of the Variational Mixture-of-Experts Autoencoder model from the paper "Variational Mixture-of-Experts Autoencoders for
Multi-Modal Deep Generative Models"
(https://arxiv.org/abs/1911.03393).

The MMVAE model uses a mixture-of-experts (MoE) aggregation. It also uses a k-samples IWAE lower bound.
The MMVAE loss writes as follows:

.. math::
    \frac{1}{M}\sum_{j=1}^{M} \mathbb E_{z^{(1)},\dots z^{(k)} \sim q_{\phi_j}(z|x_j)} \left [ \log \frac{1}{K} \sum_k \frac{p_{\theta}(z^{(k)},X)}{q_{\phi}(z|X)} \right]

The original MMVAE model uses Laplace posteriors while constraining their scaling in each direction to sum to :math:`D`,
the dimension of the latent space.

A DReG estimator can be used to compute the gradient of the IWAE loss. See (https://yugeten.github.io/posts/2020/06/elbo/)
for a nice explanation of the DReG estimator.

.. note::

    In the partially observed setting, we take the mixture of experts :math:`q_{\phi}(z|X)` over the available modalities.

    For instance, if :math:`S_{obs}(X)` is the subset of observed modalities for sample :math:`X` the loss becomes:

    .. math::
        \frac{1}{|S_{obs}(X)|}\sum_{j \in S_{obs}(X)} \mathbb E_{z^{(1)},\dots z^{(k)} \sim q_{\phi_j}(z|x_j)} \left [ \log \frac{1}{K} \sum_k \frac{p_{\theta}(z^{(k)},X)}{q_{\phi}(z|X)} \right]

    with the joint posterior :math:`q_{\phi}(z|X)` computed as the mixture of available experts.


"""

from .mmvae_config import MMVAEConfig
from .mmvae_model import MMVAE

__all__ = ["MMVAE", "MMVAEConfig"]
