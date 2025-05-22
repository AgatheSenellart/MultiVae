r"""The CRMVAE model proposed in https://openreview.net/forum?id=Rn8u4MYgeNJ.

It builds upon the MVTCAE model by adding unimodal reconstruction terms.

The joint posterior is a Product-Of-Experts :math:`q_{\phi}(z|X) \propto \prod_m q_{\phi_m}(z|x_m)`

The loss of the model then writes:

.. math::

    \mathcal{L}(X) = \sum_{m=1}^{M} \pi_m \mathbb{E}_{q_\phi(z|X)} \left( \log p_{\theta}(x_m|z) \right) + \pi_m \mathbb{E}_{q_\phi(z|x_m)} \left( \log p_{\theta}(x_m|z) \right) \\ - \sum_{m=1}^{M} \pi_m KL(q_{\phi}(z|X)||q_{\phi}(z|x_m)) - \pi_{M+1}KL(q_{\phi}(z|X)||p(z))

In practive :math:`\pi_m = \frac{1}{M+1}`.

.. note::

    This model can be used on incomplete datasets. In that case, the product of experts and the reconstructions are
    computed only available modalities for each sample.

"""

from .crmvae_config import CRMVAEConfig
from .crmvae_model import CRMVAE

__all__ = ["CRMVAEConfig", "CRMVAE"]
