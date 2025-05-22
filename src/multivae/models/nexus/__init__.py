r"""Implementation od the NEXUS model from
`Leveraging hierarchy in multimodal generative models for effective cross-modality inference
<https://www.sciencedirect.com/science/article/abs/pii/S0893608021004470>`_.

This model uses two levels of latent variables : the first level is modality-specific
:math:`z_i` ans the second level is shared :math:`z_{\sigma}`.

The diagram below illustrate the architecture of this model:

.. image:: nexus_architectures.png

The loss of the model is the sum of the bottom loss (composed of M multimodal ELBOs):

.. math::
    l_{bottom} = \sum_{i=1}^{M} \beta_i KL(q_{\phi}(z_i|x_i) || p(z_i)) - \lambda_i \mathbb{E}_{q_{\phi}(z_i|x_i)}(\log p_{\theta}(x_i|z_i))

and the top loss

.. math::
    l_{top} = \beta KL(q_{\phi}(z_{\sigma}|\bar{z}_{1::M})||p(z_{\sigma}))
    - \sum_{i=1}^{M} \gamma_i \mathbb{E}_{\substack{q_{\phi}(\bar{z}_i|x_{1::M})\\ q_{\phi}(z_{\sigma}|\bar{z}_{1:M})}}(\log(p_{\theta}(\bar{z}_i|z_{\sigma})))

The :math:`\beta, \beta_i, \lambda_i,\gamma_i` factors weighs the different terms. This model further uses annealing at the beggining of training.
The Nexus model further uses a Forced Perceptual Dropout paradigm where during training, some modalities are dropped
before computing the top loss.


.. note::
    This model can be used in the partially observed setting, by simply summing on available modalities for sample :math:`X`.

.. note::
    We didn't manage to reproduce the results presented in the paper for this model, although we followed the article
    and the official implementation closely. If you notice an error in our implementation, don't hesitate to reach out
    to us.


"""

from .nexus_config import NexusConfig
from .nexus_model import Nexus

__all__ = ["NexusConfig", "Nexus"]
