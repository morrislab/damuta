"""Damuta provides several latent variable models for mutational signature analysis

The most commonly used functions/objects are:
  - damuta.base - base classes for datasets and signatures sets
  - damuta.models - probabilstic models based on pymc3
  - damuta.plotting - useful plotting 
  - wandb.log — log metrics and media over time within your training loop
For guides and examples, see https://docs.wandb.com/guides.
For scripts and interactive notebooks, see https://github.com/wandb/examples.
For reference documentation, see https://docs.wandb.com/ref/python.
"""
__version__ = "0.0.2"

__all__ = [
    "__version__",
    "base",
    "setup",
    "save",
    "sweep",
]