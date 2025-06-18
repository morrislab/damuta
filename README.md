
<div align="center">

<p align="center"><img src="https://github.com/user-attachments/assets/4373ce68-13ee-4f8d-a1d9-229c4be8942a" width=300px /></p>

**D**irichlet **A**llocation of **MUTA**tions in cancer 

*Damage and Misrepair Signatures: Compact Representations of Pan-cancer Mutational Processes*

[![Documentation Status](https://readthedocs.org/projects/damuta/badge/?version=latest)](https://damuta.readthedocs.io/en/latest/?badge=latest) 

</div>

---

# See our [preprint](https://www.biorxiv.org/content/10.1101/2025.05.29.656360v1) for model details

![image](https://user-images.githubusercontent.com/23587234/140100948-98f10395-2bdb-4cf5-ac8b-fd66396d8d7f.png)

# DAMUTA signature definitions

* [18 Damage signatures](https://raw.githubusercontent.com/morrislab/damuta/refs/heads/main/manuscript/results/damage_sigs.csv)
* [6 Misreapair signatures](https://raw.githubusercontent.com/morrislab/damuta/refs/heads/main/manuscript/results/misreapair_sigs.csv)

nb. internally these signatures are referred to by their symbols in the graphical model: eta and phi respectively.

# Feautures

* Separately model damage and misrepair processes
* Estimate activities of DAMUTA signatures
* Fit new Damage- and Misrepair-signatures denovo

# Installation

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


# Data

Some files are omitted from this repository due to access restrictions. access can be requested from the corresponding sources: 

* [PCAWG/ICGC](https://platform.icgc-argo.org/)
* [Hartwig](https://www.hartwigmedicalfoundation.nl)
* [Genomics England](https://www.genomicsengland.co.uk/)


## Data for reproducing manuscript figures

* Unrestricted-access data and certain useful intemediate files are also available via can be downloaded from [zenodo]()

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
pcawg_cancer_types.csv | sample annotations used in [Jiao et. al](https://doi.org/10.1038/s41467-019-13825-8) | modified from [z-scores file](https://github.com/ICGC-TCGA-PanCancer/TumorType-WGS/blob/master/pcawg_mutations_types.csv)
gel_clinical_ann.csv  | tumour type annotations for 18640 samples (ICGC, HMF, GEL)| Adapted from [Degasperi et. al](https://doi.org/10.1126/science.abl9283) table S6
gel_counts.csv  | mutation type counts for 18640 samples (ICGC, HMF, GEL) | Adapted from [Degasperi et. al](https://doi.org/10.1126/science.abl9283) table S7
