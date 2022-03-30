.. damuta documentation master file, created by
   sphinx-quickstart on Tue Feb 22 17:19:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to damuta's documentation!
==================================

**Damuta** is a Python library for exploring mutational signatures in cancer. Damuta requires only mutation type counts to get started. 

You can optionally provide: 
i) Custom signature definitions
ii) Sample meta-data, such as tissue type, data source, driver mutations, and more.

.. note::

   This project is under active development.

.. automodule:: damuta
   :show-inheritance:

.. toctree::
   :maxdepth: 1
   :caption: Get started   

   installation
   examples/quickstart
   examples/index

All example notebooks can be downloaded from [github](https://github.com/morrislab/damuta/tree/pkg/docs/examples)

.. toctree::
   :maxdepth: 3
   :caption: Modules
   :hidden:
   :glob:

   damuta*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
