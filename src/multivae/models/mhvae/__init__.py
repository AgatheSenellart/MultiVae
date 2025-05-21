r"""Multimodal Hierarchical Variational Autoencoder from
    'Unified Brain MR-Ultrasound Synthesis using Multi-Modal Hierarchical Representations'
    (Dorent et al, 2O23) (https://arxiv.org/abs/2309.08747).

    The MHVAE is a hierarchical VAE that can handle multiple modalities.
    The latent variable is partitioned into disjoint groups :math:`z = \set{z_1, z_2, ..., z_L}`
    where L is the number of levels.

    The prior on the latent variables is defined as:

    .. math::
        p_{\theta}(z) = p_{\theta_L}(z_L)\prod_l p_{\theta_l}(z_l|z_{>l})

    where :math:`z_{>l}` denotes the latent variables at levels higher than l.

    The posterior is defined as :math:`q_{\phi}(z|x) = \prod_l q_{\phi_l}(z_l|x,z_{>l})`
    that approximates the intractable true posterior :math:`p_{\theta}(z|x)`.

    At each level l, the posterior :math:`q_{\phi_l}(z_l|x,z_{>l})` is approximated by a Product-of-Experts.
    At the deepest level, :math:`q_{\phi_L}(z_L|x) = p_{\theta_L}(z_L) \prod_i q_{\phi_L^{i}}(z_L|x_i)`.
    At following levels, :math:`q_{\phi_l}(z_l|x,z_{>l}) = p_{\theta_l}(z_l|z_{>l}) \prod_i q_{\phi_l^{i}}(z_l|x_i,z_{>l})`.

    Some weights are shared between the different posteriors and priors distribution.
    To allow flexibility while remaining close to the original implementation, we describe customizable
    blocks in diagram below. (adaptated from the diagram in the original paper)

.. image:: mhvae_architectures.png

.. note:: In the original paper, the authors use a discriminator loss to improve the quality of the generated samples.
    This block is not yet implemented in this version of the code.
"""

from .mhvae_config import MHVAEConfig
from .mhvae_model import MHVAE

__all__ = ["MHVAEConfig", "MHVAE"]
