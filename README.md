# damut
tandem signatures of damage and repair

clone this repo with `git clone https://github.com/harrig12/damut`

# major dependencies 

* pymc3
* numpy
* theano
* wandb
* plotly

load the env used in development with `conda env create -f .env.yml`

# package dev references

* https://python-packaging.readthedocs.io/en/latest/index.html
* https://packaging.python.org/tutorials/packaging-projects/
* https://github.com/dvav/clonosGP
* run `pytest tests -W ignore::DeprecationWarning` to run all tests with useful flag