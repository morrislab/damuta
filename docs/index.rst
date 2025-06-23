.. damuta documentation master file, created by
   sphinx-quickstart on Tue Feb 22 17:19:21 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.




DAMUTA documentation
==================================

.. image:: thumbnails/damuta_logo.png
   :width: 200px
   :align: center

| 

**DAMUTA** is a hierarchical Bayesian probabilistic model that separates DNA damage and misrepair processes.


**Key Capabilities:**

* **Damage-Misrepair Separation**: Infers distinct signatures for DNA damage processes and repair mechanisms
* **Hierarchical Modeling**: Captures sample-specific interactions and tissue-type effects  
* **Pan-Cancer Analysis**: DAMUTA signatures estimated from 18,974 whole genome sequencing samples across 23 cancer types
* **Biological Insights**: Reveals tissue-specificity patterns and DNA damage response deficiencies
* **Compact Representation**: Resolves redundancies in current signature models

| 

DAMUTA requires only mutation type counts to begin analysis. You can optionally provide:

* Custom signature definitions (damage and misrepair signatures)
* Hierarchical sample groupings for enhanced tissue-specific modeling


.. automodule:: damuta
   :show-inheritance:

.. toctree::
   :maxdepth: 1
   :caption: Get started   

   installation
   examples/quickstart
   examples/index

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
