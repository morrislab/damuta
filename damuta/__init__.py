"""Damuta provides several latent variable models for proailistic mutational signature analysis

The most commonly used functions/objects are:
  - damuta.DataSet - base classes for datasets 
  - damuta.SignatureSet - base class for signatures sets
  - damuta.models - probabilstic models based on pymc3
  - damuta.plotting - plotting functions to visualize mutational signautres and their activities
"""
__version__ = "0.0.2"

__all__ = [
    "__version__",
    "DataSet",
    "SignatureSet",
    "models",
    "plotting",
    "callbacks"
]

from damuta.base import DataSet, SignatureSet
from damuta import models as models
from damuta import plotting as plotting
from damuta import callbacks as callbacks