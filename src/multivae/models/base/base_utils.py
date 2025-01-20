import importlib

import torch
import torch.distributions as dist

model_card_template = """---
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
    return importlib.util.find_spec("huggingface_hub") is not None


def cross_entropy_(_input, _target, eps=1e-6):
    """k-Class Cross Entropy (Log Softmax + Log Loss)

    @param input: torch.Tensor (size K x bs x ...) The last dimension contains logit probabilities for each class.
    @param target: torch.Tensor (size bs x ...) The last dimension true probabilities (0 or 1) for each class.
    @param eps: error to add (default: 1e-6)
    @return loss: torch.Tensor same shape as input
    """

    _log_input = F.log_softmax(_input + eps, dim=-1)
    loss = _target * _log_input
    return loss


def cross_entropy(input, target, eps=1e-6):
    """

    Wrapper for the cross_entropy loss handling different inputs / targets types.

    """
    if isinstance(input, dict):
        if "one_hot" in input:
            _input = input["one_hot"]
        else:
            raise NotImplementedError()

    else:
        _input = input

    if isinstance(target, dict):
        if "one_hot" in target:
            _target = target["one_hot"]

        elif "tokens" in target:
            # converts to tokens proba instead of class id for text
            _target = torch.nn.functional.one_hot(target["tokens"], _input.shape[-1])
    else:
        _target = target

    return cross_entropy_(_input, _target, eps)


def set_decoder_dist(dist_name, dist_params):

    if dist_name == "normal":

        scale = dist_params.pop("scale", 1.0)
        log_prob = lambda input, target: dist.Normal(input, scale).log_prob(target)

    elif dist_name == "bernoulli":
        log_prob = lambda input, target: dist.Bernoulli(logits=input).log_prob(target)

    elif dist_name == "laplace":
        scale = dist_params.pop("scale", 1.0)
        log_prob = lambda input, target: dist.Laplace(input, scale).log_prob(target)

    elif dist_name == "categorical":
        log_prob = lambda input, target: cross_entropy(input, target)

    else:
        raise ValueError("The distribution type 'dist' is not supported")

    return log_prob
