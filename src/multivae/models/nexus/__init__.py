r"""
Implementation od the NEXUS model from 
`Leveraging hierarchy in multimodal generative models for effective cross-modality inference
<https://www.sciencedirect.com/science/article/abs/pii/S0893608021004470>`_

This model uses two levels of latent variables : the first level is modality-specific
:math:`z_i` ans the second level is shared :math:`z_{\sigma}`. 

The diagram below illustrate the architecture of this model:

.. image:: nexus_architectures.png

The loss of the model is the sum of the bottom loss (composed of M multimodal ELBOs):

.. math::
    l_{bottom} = \sum_{i=1}^{M} KL(q_{\phi}(z_i|x_i) || p(z_i)) - \mathbb{E}_{q_{\phi}(z_i|x_i)}(\log p_{\theta}(x_i|z_i))

and the top loss 

.. math::
    l_{top} = KL(q_{\phi}(z_{\sigma}|\bar{z}_{1::M})||p(z_{\sigma})) 
    - \sum_{m=1}^{M} \mathbb{E}_{\substack{q_{\phi}(\bar{z}_{1::M}|x_{1::M})\\ q_{\phi}(z_{\sigma}|\bar{z}_{1:M})}}(\log(p_{\theta}(\bar{z}_{1::M}|z_{\sigma})))

The Nexus model further uses a Forced Perceptual Dropout paradigm where during training some, modalities are dropped 
before computing the top loss. 

.. note:: 
    This model can be used in the partially observed setting. 

"""

from .nexus_config import NexusConfig
from .nexus_model import Nexus

__all__ = ["NexusConfig", "Nexus"]
