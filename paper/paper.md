---
title: 'MultiVae: A Python package for Multimodal Variational Autoencoders on Partial Datasets.'
tags:
  - Python
  - Pytorch
  - Variational Autoencoders
  - Multimodality
  - Missing data
authors:
  - name: Agathe Senellart
    orcid: 0009-0000-3176-6461
    corresponding: true
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Clément Chadebec
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2, 3"
  - name: Stéphanie Allassonnière
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1, 2, 3"
  
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
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.

header-includes:
  - \usepackage{tikz}
---

# Summary

In recent years, there has been a major boom in the development of multimodal
machine learning models. Among open topics, representation (fusion) and generation of multimodal data are very active fields of research. Recently, Multimodal
Variational Autoencoders (VAEs) have been attracting growing interest for both tasks, thanks
to their versatility, scalability, and interpretability as probabilistic latent variable
models. They are also particularly interesting in *partially observed*
settings, as most can be trained even with missing data. 
This last point makes them particularly suited for the medical field, where available datasets are often incomplete [@antelmi:2019; @aguila:poe].

We present
MultiVae, an open-source Python library bringing together unified implementations of multimodal VAEs. It has been designed
for easy and customizable use of these models on fully or partially observed data. This
library also facilitates the development and benchmarking of new algorithms by integrating
several benchmark datasets, a collection of evaluation metrics and tools for monitoring and
sharing models. 

## Multimodal Variational Autoencoders

Two main goals are commonly pursued in Multimodal Machine Learning:
(1) Learn a shared representation from multiple modalities;
(2) Learn to generate one missing modality given the ones that are available.

Multimodal Variational Autoencoders aim at solving both issues at the same time. These models learn a latent representation $z$ of all modalities in a lower dimensional common space and learn to *decode* $z$ to generate each modality.  
Let $X = (x_1, x_2, ... x_M)$ contain $M$ modalities. In the VAE setting, we define an *encoder* distribution $q_{\phi}(z|X)$ projecting the observations to the latent space, and decoders distributions $(p_{\theta}(x_i|z))_{1 \leq i \leq M}$ translating the latent code $z$ back to the observations. Those distributions are parameterized by neural networks that are trained to minimize an objective function derived from variational inference. See [@kingma] to learn more about the VAE framework and [@suzuki_survey_2022] for a survey on multimodal VAEs. 

A key differentiator of multimodal VAEs relies in the choice of the encoder $q_{\phi}(z|X)$. As illustrated in the figure below, they can be categorized into three main groups:
*Aggregated models* [@wu:2018; @shi:2019; @sutter:2021] use a mean or a product operation to aggregate the information coming from all modalities, where *Joint models* [@suzuki:2016; @vedantam:2018; @senellart:2023] use a neural network taking all modalities as input. Finally *coordinated models* [@wang_deep_2017; @tian:2019] use different latent spaces while adding a constraint term in the loss to force them to be similar. 

![Different types of multimodal VAEs \label{types_vae}](mvae_models_diagrams.png){width=100%}
We designed our library MultiVae with the aim to implement all the approaches in a unified yet modular way. 

Notably, aggregated models offer a natural way of *learning* on incomplete datasets: for an incomplete sample $X$, the encoding $z$ and the objective function can be computed using only available modalities.
However, except in our library MultiVae, there does not exist an implementation of these models that can be used on incomplete datasets in a straightforward manner. We propose a convenient way to handle missing modalities using *masks* in the loss computation of each aggregated model. 

## Data Augmentation
Another application of VAEs is Data Augmentation (DA): from sampling new latent codes $z$ and decoding them with trained models, *fully synthetic multimodal* samples can be generated to augment a dataset. 
This approach has been successfully used with unimodal VAEs to augment datasets for data-intensive deep learning applications [@chadebec_DA]. However, the use of similar sampling techniques with multimodal VAEs remains largely unexplored. 
In our library, we provide a module `multivae.samplers` with popular sampling strategies to further explore the generative abilities of these models. 

# Statement of Need

Although multimodal VAEs have interesting applications in different fields, the lack of easy-to-use and verified implementations might hinder 
applicative research. With MultiVae, we offer unified implementations, designed to be accessible even for non-specialists. In order to provide reliable implementations, we reproduced, whenever possible, a key result from the original paper. 
Related software packages have grouped together model implementations: the [Multimodal VAE Comparison Toolkit](https://github.com/gabinsane/multimodal-vae-comparison) [@sejnova:2024] includes 4 models and the [Pixyz](https://github.com/masa-su/pixyz/blob/main/examples/jmvae.ipynb)[@suzuki2023pixyz] library contains 2 multimodal models. The most closely related work, released while we were developing our library, is `multi-view-ae` [@Aguila2023], which contains a dozen of models. We compare in a summarizing table below, the different features of each work. Our library differs and complements existing software packages as follows: our API is quite different compared to previous work, the models implemented are not all the same, and for those we have in common, our implementation offers additional options. Indeed, for each model, we made sure to offer great flexibility on parameters' choices and to include all implementation details present in the original codes. Our library also offers additional features: **compatibility with incomplete data**, which we consider essential for real-life applications, **samplers** to boost the generative abilities of models, and a range of tools dedicated to research and development such **benchmark datasets** and **metrics**. We implement the most commonly used metrics in a modular way to easily evaluate any model. 


## List of Models and Features
In the table below, we list available models and features, and compare to previous work. Symbol ($\checkmark$*) indicates that the implementation includes additional options.


|Models/ Features           |Ours     |[@Aguila2023]|[@sejnova:2024]| 
|---------------------------|---------|---------|---------|
|JMVAE[@suzuki:2016]       | 	$\checkmark$* |	$\checkmark$| |
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
|DCCAE[@dccae]|| $\checkmark$||
|mWAE||	$\checkmark$||
|mmJSD[@sutter:mmjsd]||	$\checkmark$||
|gPoE[@aguila:poe]||	$\checkmark$||
|Support of Incomplete datasets|	$\checkmark$|||
|GMM Sampler|	$\checkmark$|||
|MAF Sampler, IAF Sampler|	$\checkmark$|||
|**Metrics**: {Likelihood, Coherences, FIDs, Reconstruction, Clustering}|	$\checkmark$||
|Benchmark Datasets| 	$\checkmark$||$\checkmark$||
|Model sharing via Hugging Face |	$\checkmark$|||

# Code Quality and Documentation
Our code is available on Github (https://github.com/AgatheSenellart/MultiVae) and Pypi and we provide
a full online documentation at (https://multivae.readthedocs.io/).
Our code is unit-tested with a code coverage of 94%. 
 The main features are illustrated through **tutorials** made available either as notebooks or scripts allowing users to get started easily. To further showcase how to use our library for research applications, we provide detailed *case studies* in the documentation.


# Acknowledgements

We are grateful to the authors of all the initial implementations of the models included in MultiVae. 
This work benefited from state grant managed by the Agence Nationale de la Recherche under the France 2030 program,
AN\-23-IACL-0008.
This research has been partly supported by the European Union under the (2023-2030) ERC Synergy Grant 101071601. 


# References

