import importlib

import torch
import torch.distributions as dist
import torch.nn.functional as F

MODEL_CARD_TEMPLATE = """---
language: en
tags:
- multivae
license: apache-2.0
---

### Downloading this model from the Hub
This model was trained with multivae. It can be downloaded or reloaded using the method `load_from_hf_hub`
```python
>>> from multivae.models import AutoModel
>>> model = AutoModel.load_from_hf_hub(hf_hub_path="your_hf_username/repo_name")
```
"""


def hf_hub_is_available():
    """Function to check if the huggingface_hub library is available."""
    return importlib.util.find_spec("huggingface_hub") is not None


def cross_entropy_(_input, _target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss).

    @param input: torch.Tensor (size K x bs x ...) The last dimension contains logit probabilities for each class.
    @param target: torch.Tensor (size bs x ...) The last dimension true probabilities (0 or 1) for each class.
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor same shape as input
    """
    _log_input = F.log_softmax(_input + eps, dim=-1)
    loss = _target * _log_input
    return loss


def cross_entropy(input, target, eps=1e-6):
    """Wrapper for the cross_entropy loss handling different inputs / targets types."""
    _input = input
    _target = target
    if isinstance(input, dict):
        if "one_hot" in input:
            _input = input["one_hot"]
        else:
            raise NotImplementedError()

    if isinstance(target, dict):
        if "one_hot" in target:
            _target = target["one_hot"]

        elif "tokens" in target:
            # converts to tokens proba instead of class id for text
            _target = F.one_hot(target["tokens"], _input.shape[-1])

    return cross_entropy_(_input, _target, eps)


def set_decoder_dist(dist_name, dist_params):
    """Transforms the distribution name and parameters into a callable log_prob function."""
    if dist_name == "normal":
        scale = dist_params.pop("scale", 1.0)

        def log_prob(recon, target):
            return dist.Normal(recon, scale).log_prob(target)

    elif dist_name == "bernoulli":

        def log_prob(recon, target):
            return dist.Bernoulli(logits=recon).log_prob(target)

    elif dist_name == "laplace":
        scale = dist_params.pop("scale", 1.0)

        def log_prob(recon, target):
            return dist.Laplace(recon, scale).log_prob(target)

    elif dist_name == "categorical":
        log_prob = cross_entropy

    else:
        raise ValueError("The distribution type 'dist' is not supported")

    return log_prob


def kl_divergence(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_var: torch.Tensor,
):
    r"""Compute the explicit Kullback-Leibler divergence between two gaussians.

    .. math::

        KL(p,q) = \frac{1}{2}(\log(\frac{\sigma_2²}{\sigma_1²} + \frac{\sigma_1² + (\mu_1 - \mu_2)²}{\sigma_2²} - 1)

    Args:
        mean (torch.Tensor) : mean of the first gaussian
        log_var (torch.Tensor) : log_covariance of the first gaussian
        prior_mean (torch.Tensor) : mean of the second gaussian
        prior_log_var (torch.Tensor) : log_covariance of the second gaussian

    Returns:
        torch.Tensor
    """
    kl = 0.5 * (
        prior_log_var
        - log_var
        + torch.exp(log_var - prior_log_var)
        + ((mean - prior_mean) ** 2) / torch.exp(prior_log_var)
        - 1
    )

    return kl.sum(dim=-1)


def poe(mus, logvars, eps=1e-8):
    """Compute the Product of Experts (PoE) for a list of Gaussian experts."""
    var = torch.exp(logvars) + eps
    # precision of i-th Gaussian expert at point x
    T = 1.0 / var
    pd_mu = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1.0 / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar


def stable_poe(mus, logvars):
    """Compute the Product of Experts (PoE) for a list of Gaussian experts.
    This version is more numerically stable than the naive implementation.
    """
    # If only one expert, return it
    if len(mus) == 1:
        return mus[0], logvars[0]

    # Compute ln (1/var) for each expert
    ln_inv_vars = torch.stack([-l for l in logvars])  # Compute the inverse of variances
    # ln(var_joint) = ln(1/sum(1/var)) = -ln(sum(1/var))
    ln_var = -torch.logsumexp(ln_inv_vars, dim=0)  # variances of the product of experts
    joint_mu = (torch.exp(ln_inv_vars) * mus).sum(dim=0) * torch.exp(ln_var)

    return joint_mu, ln_var


def rsample_from_gaussian(mu, log_var, N=1, return_mean=False, flatten=False):
    """Simple function to sample from a gaussian whose parameters are given by a ModelOutput.

    Args:
        mu (torch.Tensor) : mean of the gaussian
        log_var (torch.Tensor) : log_variance of the gaussian
        N(int) : number of samples to draw
        return_mean (bool): If True, each sample is the mean of the distribution.
        flatten (bool): If True, the output is flattened to be of shape (N*n_batch, *latent_dims)
    """
    sample_shape = [] if N == 1 else [N]

    if return_mean:
        z = torch.stack([mu] * N) if N > 1 else mu

    else:
        z = dist.Normal(mu, torch.exp(0.5 * log_var)).rsample(sample_shape)
    if (N > 1) and flatten:
        if len(z.shape) == 2:
            z = z.unsqueeze(0)
        z = z.reshape(-1, *z.shape[2:])

    return z
