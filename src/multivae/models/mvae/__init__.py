r"""Implementation of the MVAE model from "Multimodal Generative Models for Scalable
Weakly-Supervised Learning" (https://arxiv.org/abs/1802.05335).


The MVAE model was the first aggregated model proposed by [1].
The joint posterior is modelled as a Product-of-Experts :math:`q_{\phi}(z|X) \propto p(z)\prod_j q_{\phi_j}(z|x_j)`.
The ELBO is then optimized:

.. math::
    \mathcal{L}_{MVAE}(X) = \mathbb{E}_{q_{\phi}(z|X)}\left [\log p_{\theta}(X|z) \right] - KL(q_{\phi}(z|X)|| p_{\theta}(z))

This ELBO can be computed on a subset of modalities :math:`S` by taking only the modalities in the subset to compute the PoE:

.. math::
    q_{\phi}(z|X) \propto p(z)\prod_{j \in S} q_{\phi_j}(z|x_j)

To ensure all unimodal encoders are correctly trained, the MVAE uses a sub-sampling training paradigm, meaning that at iteration,
the ELBO is computed for several subsets: the joint subset :math:`\{1,..,M\}`, the unimodal subsets and for :math:`K` random subsets.
For each sample, the objective then becomes:

.. math::
    \mathcal{L}_{MVAE}(X) + \sum_j \mathcal{L}_{MVAE}(x_j) + \sum_{k=1}^{K} \mathcal{L}_{MVAE}((x_j)_{j \in s_k})

where :math:`s_k` are random subsets.

.. note::
    As an aggregated model, this model can be used in the partially observed setting.
    In the partially observed setting, we don't use the sub-sampling paradigm since the dataset is naturally sub-sampled,
    and for each sample :math:`X`, we compute the ELBO with only the observed modalities in :math:`S_{obs}(X)` using the posterior:

    .. math::
        q_{\phi}(z|X) \propto p(z) \prod_{j \in S_{obs}(X)} q_{\phi_j}(z|x_j).

[1] Wu et al (2018), "Multimodal Generative Models for Scalable Weakly-Supervised Learning", https://arxiv.org/abs/1802.05335

"""

from .mvae_config import MVAEConfig
from .mvae_model import MVAE

__all__ = ["MVAE", "MVAEConfig"]
