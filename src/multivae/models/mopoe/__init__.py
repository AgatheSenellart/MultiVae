r"""MoPoE model from `Generalized Multimodal ELBO (Sutter et al 2021)
<https://openreview.net/forum?id=5Y21V0RDBV>`_.

The MoPoE-VAE uses a Mixture of Product of Experts.

Formally, for each subset :math:`S \in \mathcal P(\{1 ,\dots,M\})` a PoE distribution is defined
:math:`\tilde{q}_{\phi}(z|(x_j)_{j \in S}) = PoE((q_{\phi_j}(z|x_j))_{j \in S})`.

Then the joint posterior is defined as:

.. math::
    q_{\phi}(z|X) = \frac{1}{2^M-1}\sum_{S \in  P(\{1 ,\dots,M\}, S \neq \{\} } \tilde{q}_{\phi}(z|(x_j)_{j \in S}).


The ELBO is optimized:

.. math::
    \mathcal{L}(X) = \mathbb{E}_{q_{\phi}(z|X)}\left [\ln p_{\theta}(X|z) \right] - KL(q_{\phi}(z|X) | p_{\theta}(z))

The MoPoE model can be used with additional modality-specific latent spaces.

.. note::
    To adapt this model to the partially observed setting, the loss is computed with all available subsets
    :math:`S \in  S_{obs}(X)`, where :math:`S_{obs}(X)` is the set of observed modalities for the sample :math:`X` at hand,
    instead of all subsets :math:`S \in  P(\{1 ,\dots,M\}`.


"""

from .mopoe_config import MoPoEConfig
from .mopoe_model import MoPoE
