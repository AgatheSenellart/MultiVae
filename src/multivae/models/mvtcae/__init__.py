r"""MVTCAE model from `Multi-View Representation Learning
via Total Correlation Objective <https://proceedings.neurips.cc/paper/2021/file/65a99bb7a3115fdede20da98b08a370f-Paper.pdf>`_.

MVTCAE uses a Product-of-Experts in a similar fashion as the MVAE but without the prior:

.. math::
    q_{\phi}(z|X) \sim \prod_j q_{\phi_j}(z|x_j)

The MVTCAE loss is derived from a Total Correlation Analysis and writes as follows:

.. math::
    \begin{split}
        \mathcal L(X) &= \frac{M - \alpha}{M}\mathbb{E}_{q_{\phi}(z|X)}\left [\log p_{\theta}(X|z) \right] \\&- \beta \left[(1- \alpha) KL(q_{\phi}(z|X)|| p_{\theta}(z)) + \frac{\alpha}{M} \sum_{j=1}^{M} KL(q_{\phi}(z|X) || q_{\phi_j}(z|x_j)  \right]
    \end{split}

Although this loss derives from a different analysis, it uses same terms that in the JMVAE model.
A :math:`\beta` factor weighs the regularization, while the :math:`\alpha` parameters is used to ponder the different divergence terms.

.. note::
    For the partially observed setting, we follow the authors' indications setting the variance for the missing modalities' decoders
    to :math:`\infty` which amounts to setting the reconstruction loss to 0 for those modalities.
    The KL terms for missing modalities are also set to 0.


"""

from .mvtcae_config import MVTCAEConfig
from .mvtcae_model import MVTCAE

__all__ = ["MVTCAE", "MVTCAEConfig"]
