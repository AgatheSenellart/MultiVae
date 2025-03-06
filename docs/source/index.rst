.. multivae documentation master file, created by
   sphinx-quickstart on Mon Mar  27 14:47:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*************************************
Welcome to MultiVae's documentation!
*************************************

This library implements some of the most common *Multimodal Variational Autoencoders* methods in a unifying framework for effective benchmarking and development. 
For easy benchmarking, we include ready-to-use datasets and metrics modules.
It integrates model monitoring with [Wandb](https://wandb.ai) and a quick way to save/load model from [HuggingFaceHub](https://huggingface.co/)🤗.
To improve joint generation of multimodal samples, we also propose *samplers* to explore the latent space of your model.

.. toctree::
   :caption: Basics
   :titlesonly:
   
   readme


.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :titlesonly:

   api_description
   models/multivae.models
   datasets/multivae.datasets
   metrics/multivae.metrics
   trainers/multivae.trainers
   samplers/multivae.samplers
   examples/multivae.examples
   

Setup
~~~~~~~~~~~~~

To install the latest stable release of this library run the following :

.. code-block:: bash

   $ pip install multivae

To install the latest version of this library run the following using ``pip``

.. code-block:: bash

   $ pip install git+https://github.com/AgatheSenellart/MultiVae.git 

or alternatively you can clone the github repo to access to tests, tutorials and scripts.

.. code-block:: bash

   $ git clone https://github.com/AgatheSenellart/MultiVae.git

and install the library

.. code-block:: bash

   $ cd MultiVae
   $ pip install -e .

If you clone the MultiVae's repository you will access to  the following:

- ``docs``: The folder in which the documentation can be retrieved.
- ``tests``: multivae's unit-testing using pytest.
- ``examples``: A list of ``ipynb`` tutorials and scripts describing the main functionalities of multivae.
- ``src/multivae``: The main library which can be installed with ``pip``.