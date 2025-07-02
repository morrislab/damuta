Installation
============

You can install DAMUTA using pip from PyPI or directly from GitHub.

From PyPI
---------

To install the latest stable version from PyPI, run:

.. code-block:: bash

    pip install damuta

From GitHub
-----------

To install the latest development version from GitHub, run:

.. code-block:: bash

    pip install git+https://github.com/morrislab/damuta.git

Requirements
------------

DAMUTA requires Python 3.8 or later. The main dependencies are:

- NumPy
- Pandas
- PyMC3
- Theano
- Scikit-learn

theanorc
------------

To use the GPU, ~/.theanorc should contain the following:

.. code-block:: ini

    [global]
    floatX = float64
    device = cuda

Otherwise, device will default to CPU. If complilation is slow you can increase the timeout with:

.. code-block:: ini

    [global]
    config.compile.timeout = 1000