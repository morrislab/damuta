# DAMUTA
**D**irichlet **a**llocation of **muta**tions in cancer 

[![Documentation Status](https://readthedocs.org/projects/damuta/badge/?version=latest)](https://damuta.readthedocs.io/en/latest/?badge=latest)
      

![image](https://user-images.githubusercontent.com/23587234/140100948-98f10395-2bdb-4cf5-ac8b-fd66396d8d7f.png)


# install

DAMUTA is built on pymc3 - which depends on theano. To use theano with gpu, you will need to install pygpu. The simplest way to do so is via conda.

`conda create -n damuta -c conda-forge python=3.8 pygpu=0.7.6`

## from pipy

DAMUTA is available on [pipy test server](https://test.pypi.org/project/damuta/)

## from github

Clone this repo `git clone https://github.com/morrislab/damuta`
Install requirements `pip install -r damuta/requirements.txt`
Install damuta `pip install -e damuta`


# theanorc

To use the GPU, `~/.theanorc` should contain the following:

```
[global]
floatX = float64
device = cuda
```

Otherwise, device will default to CPU. 

