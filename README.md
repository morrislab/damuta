# DAMUTA
**D**irichlet **A**llocation of **muta**tions in cancer 

![image](https://user-images.githubusercontent.com/23587234/140100948-98f10395-2bdb-4cf5-ac8b-fd66396d8d7f.png)


# major dependencies 

* pymc3
* numpy
* theano
* wandb
* plotly

load the env used in development with `conda env create -f .env.yml`

# theanorc

To use the GPU, `~/.theanorc` should contain the following:

```
[global]
floatX = float64
device = cuda
```

Otherwise, device will default to CPU. 

# package dev references

* https://python-packaging.readthedocs.io/en/latest/index.html
* https://packaging.python.org/tutorials/packaging-projects/
* https://github.com/dvav/clonosGP
* run `pytest tests -W ignore::DeprecationWarning` to run all tests with useful flag
