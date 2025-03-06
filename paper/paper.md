---
title: 'MultiVae: A Python package for Multimodal Variational Autoencoders on Partial Datasets '
tags:
  - Python
  - Pytorch
  - Variational Autoencoders
  - multimodality
  - missing data
authors:
  - name: Agathe Senellart
    orcid: 0000-0000-0000-0000
    corresponding: true
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Stéphanie Allassonnière
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  
affiliations:
 - name: Université de Paris-Cité
   index: 1
 - name: Inria
   index: 2
 - name: Inserm
   index: 3
date: 3 March 2025
bibliography: [./paper.bib]

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.

header-includes:
  - \usepackage{tikz}
---

# Summary

In recent years, there has been a major boom in the development of multimodal
machine learning models. Among open topics, representation (fusion) and generation of multimodal data are very active fields of research. Recently, Multimodal
Variational Autoencoders (VAEs) have been attracting growing interest for both tasks, thanks
to their versatility, scalability, and interpretability as probabilistic latent variable
models. They are also particularly interesting models in the *partially observed*
setting, as most of them can learn even with missing data. 
This last point makes them particularly interesting for research fields such as the medical field, where missing data are commonplace [@antelmi:2019; @aguila:poe].

We present
MultiVae, an open-source Python library for bringing together unified implementations of multimodal VAEs. It has been designed
for easy, customizable use of these models on fully or partially observed data. This
library also facilitates the development and benchmarking of new algorithms by integrating
several benchmark datasets, a variety of evaluation metrics and tools for monitoring and
sharing models. 

## Multimodal Variational Autoencoders

In Multimodal Machine Learning, two goals are generally targeted:
(1) Learn a shared representation from multiple modalities;
(2) Learn to generate one missing modality given the ones that are available.

Multimodal Variational Autoencoders aim at solving both issues at the same time. These models learn a latent representation $z$ of all modalities in a lower dimensional common space and learn to *decode* $z$ to generate any modality [@suzuki_survey_2022].  
Let $X = (x_1, x_2, ... x_M)$ contain $M$ modalities. In the VAE setting, we suppose that the generative process behind the observed data is the following:
\begin{align}
&z \sim p(z)
& \forall 1 \leq i \leq M, x_i|z \sim p_{\theta}(x_i|z)
\end{align}
where $p(z)$ is a prior distribution that is often fixed, and $p_{\theta}(x_i|z)$ are called *decoders* and are parameterized by neural network. 
Typically, $p_{\theta}(x_i|z) = \mathcal{N}(x_i; \mu_{\theta}(z), \sigma_{\theta}(z))$ where $\mu_{\theta}, \sigma_{\theta}$ are neural networks.
We aim to learn these *decoders* that translate $z$ into the high dimensional data $x_i$. At the same time, we aim to learn an *encoder* $q_{\phi}(z|X)$ that map observations to the latent space. $q_{\phi}(z|X)$ is also parameterized by a neural network. 
Derived from variational inference [@kingma], the VAE objective writes:
$$\mathcal{L}(X) =  \mathbb{E}_{q_\phi(z|X)}\left( \sum_i \ln(p_{\theta}(x_i|z)) \right) - KL(q_{\phi}(z|X)|p(z))$$

The first term is a reconstruction loss and the second term can be seen as a regularization term that avoids overfitting. A typical training of a multimodal VAE consists in encoding the data with the encoder, reconstructing each modality with the decoders and taking a gradient step to optimize the loss $\mathcal{L}(X)$. 

Most multimodal VAEs differ in how they construct the encoder $q_{\phi}(z|X)$. In the figure below, we summarize several approaches:
*Aggregated models* [@wu:2018; @shi:2019; @sutter:2021] use a mean or a product operation to aggregate the information coming from all modalities, where *Joint models* [@suzuki:2016; @vedantam:2018; @senellart:2023] use a neural network taking all modalities as input. Finally *coordinated models* [@dvcca; @tian:2019] use different latent spaces but add a constraint term in the loss to force them to be similar. 

![Different types of multimodal VAEs \label{types_vae}](mvae_models_diagrams.png){width=100%}
<!-- 
Recent extensions of multimodal VAEs include additional terms to the loss, or use multiple [@palumbo_mmvae_2023] or hierarchical [@vasco2022leveraging; @Dorent_2023] latent spaces to more comprehensively describe the multimodal data.  -->
In our library, we implement all these approaches in an unified and modular way.

Aggregated models offer a natural way of learning on incomplete datasets: for an incomplete sample $X$, we use only the available modalities to encode the data and compute the loss. However, except in MultiVae, there doesn't exist an implementation of these models that can be used on incomplete datasets in a straightforward manner. 

## Data Augmentation
Another application of these models is Data Augmentation (DA): from sampling latent codes $z$ and decoding them, *fully synthetic multimodal* samples can be generated to augment a dataset. DA has been proven useful in many data-intensive deep learning applications [@chadebec_DA]. In a dedicated module `multivae.samplers`, we propose different ways of sampling latent codes $z$ to further explore the generative abilities of these models. 

# Statement of need

Although multimodal VAEs have interesting applications in different fields, the lack of easy-to-use and verified implementations might hinder 
applicative research. With MultiVae, we offer unified implementations, designed to be easy to use by non-specialists and even on incomplete data. In order to propose reliable implementations of each method, we tried to reproduce, whenever possible, a key result from the original paper. 
Some works similar to ours have grouped together model implementations: the [Multimodal VAE Comparison Toolkit](https://github.com/gabinsane/multimodal-vae-comparison) [@sejnova:2024] includes 4 models and the [Pixyz](https://github.com/masa-su/pixyz/blob/main/examples/jmvae.ipynb)[@suzuki2023pixyz] library contains 2 multimodal models. The work closest to ours and released while we were developping our library is `multi-view-ae` [@Aguila2023], which contains a dozen of models. We compare in a summarizing table below, the different features of each work.  Our library complements what already exists: our API is quite different compared to previous work, the models implemented are not all the same, and for those we have in common, our implementation offers additional parameterization options. Indeed, for each model, we've made sure to offer great flexibility on parameters and to include all implementation details present in the original codes. Our library also offers additional features: **compatibility with incomplete data**, which we consider essential for real-life applications, and a range of tools dedicated to research and development of new algorithms: benchmark datasets, metrics modules and samplers, for testing and analyzing models. 
<!-- Our library also supports distributed training and straightforward model sharing via HuggingFace Hub[@huggingface].  -->
<!-- Therefore our work complements existing options and addresses different needs.  -->

## List of models and features
In the Table below, we list available models and features, and compare to previous work. This symbol ($\checkmark$*) indicates that the implementation include additional options.


|Models/ Features           |Ours     |[@Aguila2023]|[@sejnova:2024]| 
|---------------------------|---------|---------|---------|
|JMVAE [@suzuki:2016]       | 	$\checkmark$* |	$\checkmark$| |
|MVAE[@wu:2018]             | 	$\checkmark$*|	$\checkmark$|$\checkmark$|
|MMVAE[@shi:2019]           |	$\checkmark$*|	$\checkmark$|$\checkmark$|
|MoPoE[@sutter:2021]        |	$\checkmark$*|	$\checkmark$|$\checkmark$|
|DMVAE[@lee:2021]           |	$\checkmark$|	$\checkmark$*|$\checkmark$|
|MVTCAE[@hwang2021multi]    |	$\checkmark$|	$\checkmark$||
|MMVAE+[@palumbo_mmvae_2023]|	$\checkmark$*|	$\checkmark$||
|CMVAE[@palumbo2024deep]    |	$\checkmark$|||
|Nexus[@vasco2022leveraging]|	$\checkmark$|||
|CVAE[@kingma]              |	$\checkmark$|||
|MHVAE[@dorent:2023]        |	$\checkmark$|||
|TELBO[@vedantam:2018]      |	$\checkmark$|||
|JNF[@senellart:2023]       |	$\checkmark$|||
|CRMVAE[@suzuki:2023:mitigating]|$\checkmark$|||
|MCVAE[@antelmi:2019]||	$\checkmark$||
|mAAE||	$\checkmark$||
|DVCCA[@wang_deep_2017]||	$\checkmark$||
|mWAE||	$\checkmark$||
|mmJSD[@sutter:mmjsd]||	$\checkmark$||
|gPoE[@aguila:poe]||	$\checkmark$||
|Support of Incomplete datasets|	$\checkmark$|||
|GMM Sampler|	$\checkmark$|||
|MAF Sampler, IAF Sampler|	$\checkmark$|||
|**Metrics**: Likelihood, Coherences, FIDs, Reconstruction, Clustering|	$\checkmark$||
|Ready-to-use Datasets| 	$\checkmark$||$\checkmark$||
|Model sharing via Hugging Face |	$\checkmark$|||

An important difference in our user-interface, is that we handle all training and model parameters within python dataclasses while [@aguila:2023 ; @sejnova:2024] uses independant `YAML` configuration files.


# Code quality and documentation
Our code is available on Github (https://github.com/AgatheSenellart/MultiVae) and Pypi and we provide
a full online documentation at (https://multivae.readthedocs.io/en/latest/). The main features are illustrated through tutorials made available either
as notebooks or scripts allowing users to get started easily. To further showcase how to use our library for research applications, we provide detailed case studies [here](https://multivae.readthedocs.io/en/latest/examples/multivae.examples.html).


# Acknowledgements

We are grateful to the authors of all the initial implementations of the models included in MultiVae. 

# References

