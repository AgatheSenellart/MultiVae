.. multivae documentation master file, created by
   sphinx-quickstart on Mon Mar  27 14:47:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**********************************
Welcome to multivae's documentation!
**********************************

This library gathers some of the most common multi-modal Variational AutoEncoder (VAE)
implementation in PyTorch. 


.. toctree::
   :maxdepth: 1
   :caption: multivae
   :titlesonly:

   models/multivae.models
   datasets/multivae.datasets
   metrics/multivae.metrics
   trainers/multivae.trainers

Setup
~~~~~~~~~~~~~

To install the latest stable release of this library run the following using ``pip``

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
- ``examples``: A list of ``ipynb`` tutorials and script describing the main functionalities of multivae.
- ``src/multivae``: The main library which can be installed with ``pip``.