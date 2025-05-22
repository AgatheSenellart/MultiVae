r"""Implementation of "MMVAE+: Enhancing the Generative Quality of Multimodal VAEs without Compromises"
(https://openreview.net/forum?id=sdQGxouELX).

The MMVAE+ model is an aggregated model that uses
multiple latent spaces : :math:`z` is the latent code *shared* accross modalities and :math:`w_j` is private latent code of modality :math:`j \in [|1 , M|]`.

It also uses auxiliary prior distributions for each private latent spaces :math:`r_j(w_j)` with a scale parameter that is learned.

As for the MMVAE, the joint posterior for the shared latent code is a Mixture-Of-Experts of the unimodal posteriors:

.. math::
    q_{\phi_z}(z|X) = \frac{1}{M} \sum_{m =1}^{M} q_{\phi_{z_m}}(z|x_m)

The loss of the MMVAE+ model then writes as follows:

.. math::
    \frac{1}{M}\sum_{m=1}^{M} \mathbb E_{
    \substack{
        z_m^{1::K} \sim q_{\phi_{z_m}}(z|x_m)\\w_m^{1::K} \sim q_{\phi_{w_m}}(w_m|x_m) \\ \tilde{w}_{n\neq m}^{1::K} \sim r_n(w_n)
    }
    } \log \frac{1}{K} \sum_{k=1}^{K} D^{\beta}_{\Phi,\Theta}(X,z^k, \tilde{w}_1^k, \tilde{w}_2^k,.., w_m^k, .., \tilde{w}_M^k)

with

.. math::
    D^{\beta}_{\Phi,\Theta}(X,z^k, \tilde{w}_1^k, \tilde{w}_2^k,.., w_m^k, .., \tilde{w}_M^k) = \frac{p_{\theta_m}(x_m|z^k, w_m^k)(p(z^k)p(w_m^k))^{\beta}}{(q_{\phi_z}(z^k|X)q_{\phi_{w_m}}(w_m^k|x_m))^{\beta}}\prod_{n \neq m}p_{\theta_n}(x_n|z^k,\tilde{w}_n^k)

It uses a K-importance sampled estimator of the likelihood and a :math:`\beta` factor that can be tuned to promote disentanglement in the latent space.
In this objective function, the modality private information :math:`w_m` is only used for self reconstruction and not for cross-modal generation.
For crossmodal generation, the shared semantic content flows through the shared latent variable :math:`z`.

.. note::
    For the partially observed case, that loss can be computed using only available sample instead of all modalities.

    If we only observe modalities in :math:`S_{obs}(X)` the loss for sample :math:`X` becomes:

    .. math::
        \frac{1}{|S_{obs}(X)|}\sum_{m \in S_{obs}(X)} \mathbb E_{
        \substack{
            z_m^{1::K} \sim q_{\phi_{z_m}}(z|x_m)\\w_m^{1::K} \sim q_{\phi_{w_m}}(z|w_m) \\ \tilde{w}_{n\neq m}^{1::K} \sim r_n(w_n)
        }
        } \log \frac{1}{K} \sum_{k=1}^{K} D^{\beta}_{\Phi,\Theta}(X,z^k, \tilde{w}_1^k, \tilde{w}_2^k,.., w_2^m, .., \tilde{w}_M^k)

    where:

    .. math::
    D^{\beta}_{\Phi,\Theta}(X,z^k, \tilde{w}_1^k, \tilde{w}_2^k,.., w_m^k, .., \tilde{w}_M^k) = \frac{p_{\theta_m}(x_m|z^k, w_m^k)(p(z^k)p(w_m^k))^{\beta}}{(q_{\phi_z}(z^k|X)q_{\phi_{w_m}}(w_m^k|x_m))^{\beta}}\prod_{n in S_{obs}(X), n\neq m}p_{\theta_n}(x_n|z^k,\tilde{w}_n^k)

    In simpler terms; we reconstruct only available modalities and compute the joint posterior with available modalities only.

    .. math::
        q_{\phi_z}(z|X) = \frac{1}{|S_{obs}(X)|} \sum_{m \in S_{obs}(X)} q_{\phi_{z_m}}(z|x_m)
"""

from .mmvaePlus_config import MMVAEPlusConfig
from .mmvaePlus_model import MMVAEPlus

__all__ = ["MMVAEPlus", "MMVAEPlusConfig"]
