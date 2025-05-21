r"""Joint Normalizing Flows (JNF) from https://arxiv.org/abs/2502.03952.

JNF uses a joint encoder to model :math:`q_{\phi}(z|X)` and surrogate unimodal encoders :math:`q_{\phi_j}(z|x_j)` for :math:`1\leq j\leq M`.

The loss used is the same as the JMVAE (with :math:`\alpha = 1`) but the unimodal encoders
:math:`q_{\phi_j}(z|x_j)` are modelled with Masked Autoregressive Flows.

.. math::
    \mathcal{L}_{JNF}(X) = \mathbb E_{q_{\phi}(z|X)}\left[ \frac{p_{\theta}(z,X)}{q_{\phi}(z|X)} \right] - \sum_{j=1}^{M} KL(q_{\phi}(z|X) || q_{\phi_j}(z|x_j) )

Contrary to the JMVAE, this model is trained with separate stages: : first the joint encoder is trained, then the unimodal encoders are trained.

.. note::
    This model must be trained with ~multivae.trainers.multistage_trainer.MultiStageTrainer.

.. note::
    As it uses a joint encoder, this model can not be used in the partially observed setting.
"""

from .jnf_config import JNFConfig
from .jnf_model import JNF

__all__ = ["JNFConfig", "JNF"]
