r"""Conditional Variational Autoencoder model (https://arxiv.org/abs/1906.02691).

This model is used to model the distribution of :math:`y` knowing :math:`x`.
The general generative model is:

.. math::
    p_{\theta}(y,z|x) = p_{\theta}(y|z,x)p_{\theta}(z|x)

where :math:`p_{\theta}(y|z,x)` is called the decoder and :math:`p_{\theta}(z|x)` is a prior distribution.
This prior might depend on :math:`x` or not, if :math:`z` is considered independant of :math:`x`.
The approximate posterior distribution can also depend on :math:`x` : :math:`q_{\phi}(z|y,x)` (or not).

The Evidence Lower Bound writes:

.. math::
    \mathcal{L(y|x)} = \mathbb{E}_{q_{\phi}(z|y,x)}\left( \ln p_{\theta}(y|z,x) \right) - KL(q_{\phi}(z|y,x)||p_{\theta}(z|x))






"""

from .cvae_config import CVAEConfig
from .cvae_model import CVAE

__all__ = ["CVAEConfig", "CVAE"]
