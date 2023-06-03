![logo](./static/multivae_logog.png)


<!-- Add buttons here -->
[![codecov](https://codecov.io/gh/clementchadebec/multimodal_vaes/branch/main/graph/badge.svg?token=0077GYjHKo)](https://codecov.io/gh/clementchadebec/multimodal_vaes)



<!-- Describe your project in brief -->

This library implements some of the most common *Multimodal Variational Autoencoders* methods in a unifying framework for effective benchmarking and development. You can find the list of implemented models below.
It includes ready to use datasets like MnistSvhn 🔢, CelebA 😎 and PolyMNIST, 
and the most used metrics : Coherences, Likelihoods and FID. 
It integrates model monitoring with [Wandb](https://wandb.ai) and a quick way to save/load model from [HuggingFaceHub](https://huggingface.co/)🤗.


# Implemented models

|Model|Paper|Official Implementation|
|:---:|:----:|:---------------------:|
|JMVAE|[Joint Multimodal Learning with Deep Generative Models](https://arxiv.org/abs/1611.01891)|[link](https://github.com/masa-su/jmvae)|
|MVAE| [Multimodal Generative Models for Scalable Weakly-Supervised Learning](https://proceedings.neurips.cc/paper/2018/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html)|[link](https://github.com/mhw32/multimodal-vae-public)|
|MMVAE|[Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models](https://proceedings.neurips.cc/paper/2019/hash/0ae775a8cb3b499ad1fca944e6f5c836-Abstract.html)|[link](https://github.com/iffsid/mmvae)|
|MoPoE| [Generalized Multimodal ELBO](https://openreview.net/forum?id=5Y21V0RDBV)|[link](https://github.com/thomassutter/MoPoE)|
|MVTCAE | [Multi-View Representation Learning via Total Correlation Objective](https://proceedings.neurips.cc/paper/2021/hash/65a99bb7a3115fdede20da98b08a370f-Abstract.html)|[link](https://github.com/gr8joo/MVTCAE/)|
|JNF,JNF-DCCA| Improving Multimodal Joint Variational Autoencoders through Normalizing Flows and Correlation Analysis | link|
|MMVAE + |MMVAE+: ENHANCING THE GENERATIVE QUALITY OF MULTIMODAL VAES WITHOUT COMPROMISES | [link](https://openreview.net/forum?id=sdQGxouELX)|

# Quickstart

<!-- Add a demo for your project -->
Install the library by running 
```shell
git clone https://github.com/AgatheSenellart/multimodal_vaes.git
cd multimodal_vaes
pip install .
```
Load a dataset easily :
```python
from multivae.data.datasets import MnistSvhn
train_set = train_set = MnistSvhn(data_path = 'your_data_path',split="train", download=True)

```
Instantiate your favorite model :
```python
from multivae.models import MVTCAE, MVTCAEConfig
model_config = MVTCAEConfig(
    latent_dim=20, 
    input_dims = {'mnist' : (1,28,28),'svhn' : (3,32,32)})
model = MVTCAE(model_config)

```
Define a trainer and train the model !
```python
training_config = BaseTrainerConfig(
    learning_rate=1e-3,
    num_epochs=30
)

trainer = BaseTrainer(
    model=model,
    train_dataset=train_set,
    training_config=training_config,
)
trainer.train()
```

# Table of Contents


- [Models available](#implemented-models)
- [Quickstart](#quickstart)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Contribute](#contribute)
- [Reproducibility statement](#reproducibility-statement)
- [License](#license)

# Installation
[(Back to top)](#table-of-contents)


```shell
git clone https://github.com/AgatheSenellart/multimodal_vaes.git
cd multimodal_vaes
pip install .
```

# Usage
[(Back to top)](#table-of-contents)

Our library allows you to use any of the models with custom configuration, encoders and decoders architectures and datasets easily. 
See our tutorial Notebook at /examples/tutorial_notebooks/getting_started.ipynb to easily get the gist of principal features. 


# Contribute
[(Back to top)](#table-of-contents)

If you want to contribute by adding models to the library, clone the repository and install it in editable mode by using the -e option
```shell
pip install -e .
```


# Reproducibility statement

All implemented models are validated by reproducing a key result of the paper. 



# License
[(Back to top)](#table-of-contents)

