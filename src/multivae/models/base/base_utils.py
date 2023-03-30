import importlib

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
