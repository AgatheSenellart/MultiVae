r"""Implementation of the Joint Multimodal VAE model from the paper "Joint Multimodal Learning with Deep
Generative Models" (http://arxiv.org/abs/1611.01891).

The JMVAE model is one of the first multimodal variational autoencoders models.
It has a dedicated joint encoder network :math:`q_{\phi}(z|X)` and surrogate unimodal encoders
:math:`q_{\phi_j}(z|x_j)`. The JMVAE loss has additional terms to the ELBO to fit the unimodal encoders:

.. math::
    \mathcal{L}_{JMVAE}(X) = \mathbb E_{q_{\phi}(z|X)}\left[ p_{\theta}(z|X) \right] - KL\left(q_{\phi}(z|X)||p_{\theta}(z)\right) - \alpha \sum_{j=1}^{M} KL \left (q_{\phi}(z|X) || q_{\phi_j}(z|x_j) \right)

where :math:`M` is the number of modalities.
This loss can be linked to the Variation of Information (VI) between modalities [1].
:math:`\alpha` is the parameter that controls a trade-off between the quality of reconstruction and the quality of cross-modal generation
[1].
This model has been proposed for only two-modalities, but an extension has been proposed in [2] for additional modalities.

During inference, when :math:`M \leq 2`, the subset posteriors :math:`p_{\theta}(z|(x_j)_{j \in S})` can be approximated by
the product of experts (PoE)
of the already trained unimodal encoders :math:`q_{\phi}(z|x_j)_{1 \leq j \leq M}`.
Since the unimodal posteriors are normal distributions, the PoE has a closed-form and can easily be computed.

The JMVAE model uses annealing during training: which means that a weighting factor that ponders the regularizations terms is linearly augmented from 0 to 1 during the first epochs.

.. note::
    As it uses a joint encoder network, this model can not be trained with partially observed samples.

[1] Suzuki et al, 2016. Joint Multimodal Learning with Deep Generative Models.

[2] Senellart et al, 2023. Improving Multimodal Variational Autoencoders with Normalizing Flows and Deep Canonical Correlation Analysis.


"""

from .jmvae_config import JMVAEConfig
from .jmvae_model import JMVAE

__all__ = ["JMVAEConfig", "JMVAE"]
