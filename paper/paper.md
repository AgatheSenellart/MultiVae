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
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

In recent years, there has been a major boom in the development of multimodal
machine learning models. Among open topics, representation (fusion) and genera-
tion of multimodal data are very active fields of research. Recently, Multimodal
Variational Autoencoders have been attracting growing interest for both tasks, thanks
to their versatility, scalability, and interpretability as probabilistic latent variable
models. They are also particularly interesting models in the partially observed
setting, as most of them can learn even with missing data. 
Develop on the importance of this setting that is the standard one in medical applications for instance.
In this article, we present
MultiVae, an open-source Python library designed to bring together unified imple-
mentations of multimodal generative autoencoders models. It has been designed
for easy, customizable use of these models on partially or fully observed data. This
library facilitates the development and benchmarking of algorithms by integrating
several popular datasets, variety of evaluation metrics and tools for monitoring and
sharing models. 

# Statement of need

- `MultiVae` is Pytorch based and share the same structure as Pythae

- It was designed to be easy to use even without a lot of knowledge about Multimodal VAEs to make accessible to people from different research fields

- It includes a lot of benchmarking tools as datasets, metrics, samplers to make development and analysis of new multimodal VAEs models easier. 

- Most implementations are verified by reproducing a key result of the original paper. We include a reproducibilty statement 
  with our results. 

- All models present with a lot of options and includes all coding tricks originally used by the authors to get the best results out of a model.

- We focused on making the models easily usable on real world datasets which means adapting them for incomplete datasets.

Related works:

multiview-ae presents with a different design (all configurations are handled with .yaml files where as for us, all training/ model config are handled with python dataclasses) / not exactly the same models / not adapted for partial data / we have additional features such as  metrics/ loading on hf, samplers, etc ...
We believe our work complements theirs


What should I include below:

- Quick review of multimodal VAEs, what is does, what are the different types of models, etc...

- A more detailed description of all functionalities and structural diagram

- A case study on partial data ? 

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References