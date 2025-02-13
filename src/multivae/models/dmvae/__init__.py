"""
Implementation of the DMVAE model from

"Private-Shared Disentangled Multimodal VAE for Learning of Latent
Representations" (Lee & Pavlovic 2021)

(https://par.nsf.gov/servlets/purl/10297662)

"""

from .dmvae_config import DMVAEConfig
from .dmvae_model import DMVAE

__all__ = ["DMVAE", "DMVAEConfig"]
