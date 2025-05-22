r"""Implementation the TELBO algorithm from "Generative Models of Visually Grounded Imagination"
(https://arxiv.org/abs/1705.10762).


The TELBO model use a joint encoder :math:`q_{\phi}(z|X)` as the JMVAE but uses the following Triple ELBO loss:

.. math::
    \mathcal L(X) = \mathbb E_{q_{\phi}(z|X)}\left[ \frac{p_{\theta}(z,X)}{q_{\phi}(z|X)} \right] + \sum_{j=1}^{M} \mathbb E_{q_{\phi}(z|x_j)}\left[ \frac{p_{\theta}(z,x_j)}{q_{\phi}(z|x_j)} \right]


It is trained with a two-steps training, first learning the joint encoder and decoders then
training the unimodal encoders :math:`q_{\phi}(z|x_j)` with previous parameters fixed.

.. note::
    This model must be trained with the ~multivae.trainers.multistage.MultiStageTrainer

.. note::
    As it uses a joint encoder network, this model can not be trained with partially observed samples.


"""

from .telbo_config import TELBOConfig
from .telbo_model import TELBO

__all__ = ["TELBOConfig", "TELBO"]
