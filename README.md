
<div align="center">

<p align="center"><img src="https://github.com/user-attachments/assets/4373ce68-13ee-4f8d-a1d9-229c4be8942a" width=300px /></p>

**D**irichlet **A**llocation of **MUTA**tions in cancer 

*Damage and Misrepair Signatures: Compact Representations of Pan-cancer Mutational Processes*

[![Documentation Status](https://readthedocs.org/projects/damuta/badge/?version=latest)](https://damuta.readthedocs.io/en/latest/?badge=latest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![PyPI version](https://badge.fury.io/py/damuta.svg)](https://badge.fury.io/py/damuta) 

</div>

---


# Features

* Separately model damage and misrepair processes
* Estimate activities of DAMUTA signatures
* Fit new Damage- and Misrepair-signatures denovo

## DAMUTA signature definitions

* [18 Damage signatures](https://raw.githubusercontent.com/morrislab/damuta/refs/heads/main/docs/examples/example_data/damage_signatures.csv)
* [6 Misrepair signatures](https://raw.githubusercontent.com/morrislab/damuta/refs/heads/main/docs/examples/example_data/misrepair_signatures.csv)

nb. internally these signatures are referred to by their symbols in the graphical model: eta and phi respectively.

## Model

![image](https://user-images.githubusercontent.com/23587234/140100948-98f10395-2bdb-4cf5-ac8b-fd66396d8d7f.png)

# Repo contents

- [damuta](./damuta) python package
- [docs](./docs) package documentation
- [example notebooks](./docs/examples/) usage examples & tutorials
- [manuscript code](./manuscript) code to reproduce experiments and plots reported in manuscript

# Documention

DAMUTA documentation is hosted via [readthedocs](https://damuta.readthedocs.io/en/latest/)


# System requirements

Recommended Requirements:

* Unix-based system (tested on CentOS Linux 7)
* RAM: 16+ GB (for large-scale pan-cancer datasets)
* CPU: 4+ cores, 3.0+ GHz/core
* GPU: NVIDIA GPU with CUDA support (optional, for GPU acceleration via Theano)


Software dependencies are specified in (damuta_env.yml)[./damuta_env.yml]

# Installation

## From github

Clone this repo `git clone https://github.com/morrislab/damuta`


```bash
conda env create -f damuta_env.yml
conda activate damuta
pip install -e .
```

## from PyPI

```bash
pip install damuta
```


## theanorc

To use the GPU, `~/.theanorc` should contain the following:

```
[global]
floatX = float64
device = cuda
```

Otherwise, device will default to CPU. 

## Installation Time Estimates

Conda environment setup: ~3-5 minutes
Package installation: ~2-3 minutes


# Demo

See [quickstart](https://damuta.readthedocs.io/en/latest/examples/quickstart.html) to get started (~5 min)

# Reproducing manuscript results

See [manuscript code](./manuscript) for code to reproduce experiments and plots reported in manuscript

Some data files are omitted from this repository due to access restrictions. Access can be requested from the corresponding sources: 

* [PCAWG/ICGC](https://platform.icgc-argo.org/)
* [Hartwig](https://www.hartwigmedicalfoundation.nl)
* [Genomics England](https://www.genomicsengland.co.uk/)


## Data for reproducing manuscript figures

Unrestricted-access data and certain useful intemediate files are also available via can be downloaded from [zenodo](https://zenodo.org/records/15685052)

To download and organize these data:

```
# in top-level directory
wget  https://zenodo.org/records/15685052/files/damuta_zenodo.zip
unzip damuta_zenodo

mv damuta_zenodo/data/* manuscript/data
mv damuta_zenodo/figure_data/* manuscript/results/figure_data

# clean up now-empty directories
rmdir damuta_zenodo/data damuta_zenodo/figure_data damuta_zenodo
```

## Some useful public data

file name | info |  source  
---       |  ---                 | --- 
COSMIC_v3.2_SBS_GRCh37.csv | [COSMIC database](https://cancer.sanger.ac.uk/signatures/downloads/)
icgc_sample_annotations_summary_table.txt | sample annotations used by PCAWG heterogeneity & evolution working group | [ICGC data portal](https://dcc.icgc.org/releases/PCAWG/evolution_and_heterogeneity)
PCAWG_sigProfiler_SBS_signatures_in_samples | counts of mutations attributed to each signature for PCAWG samples | [syn11738669.7](https://www.synapse.org/#!Synapse:syn11738669.7)
pcawg_counts.csv | mutation type counts in PCAWG samples | Derived from [syn7357330](https://www.synapse.org/#!Synapse:syn7357330)
pcawg_cancer_types.csv | sample annotations used in [Jiao et. al](https://doi.org/10.1038/s41467-019-13825-8) | Adapted from [z-scores file](https://github.com/ICGC-TCGA-PanCancer/TumorType-WGS/blob/master/pcawg_mutations_types.csv)
gel_clinical_ann.csv  | tumour type annotations for 18640 samples (ICGC, HMF, GEL)| Adapted from [Degasperi et. al](https://doi.org/10.1126/science.abl9283) table S6
gel_counts.csv  | mutation type counts for 18640 samples (ICGC, HMF, GEL) | Adapted from [Degasperi et. al](https://doi.org/10.1126/science.abl9283) table S7

## Citation

```

@misc{harrigan_damage_2025,
	title = {Damage and {Misrepair} {Signatures}: {Compact} {Representations} of {Pan}-cancer {Mutational} {Processes}},
	copyright = {© 2025, Posted by Cold Spring Harbor Laboratory. This pre-print is available under a Creative Commons License (Attribution-NonCommercial 4.0 International), CC BY-NC 4.0, as described at http://creativecommons.org/licenses/by-nc/4.0/},
	shorttitle = {Damage and {Misrepair} {Signatures}},
	url = {https://www.biorxiv.org/content/10.1101/2025.05.29.656360v1},
	doi = {10.1101/2025.05.29.656360},
	language = {en},
	urldate = {2025-06-02},
	publisher = {bioRxiv},
	author = {Harrigan, Caitlin F. and Campbell, Kieran and Morris, Quaid and Funnell, Tyler},
	month = jun,
	year = {2025},
}

```
