There are a few different types of data file in this folder:

## signature definitions

These are (n signatures) x (96) matrices that define the distribution over mutaiton types of each signature. Columns should sum to 1.

file name | mutation type format | number of signatures |  source  
---       |  ---          | ---                  | --- 
COSMIC_v3.2_SBS_GRCh37.csv | type | 79 | [COSMIC database](https://cancer.sanger.ac.uk/signatures/downloads/)
sigProfiler_SBS_signatures_2019_05_22.csv | type/subtype | 68 | [syn11738319](https://www.synapse.org/#!Synapse:syn11738319)
pcawg_localsigs.csv | type/subtype | 222 | modified from [syn11853232](https://www.synapse.org/#!Synapse:syn11853232)
degasperi_refsigs.csv | type/subtype | 41 | modified from [Degasperi et. al 2020](https://doi.org/10.1038/s43018-020-0027-5) supplementary table 4
degasperi_localsigs.csv | type | 192 | [Degasperi et. al 2020](https://doi.org/10.1038/s43018-020-0027-5) supplementary table 2


## tumour meta data

file name | contents |  source  
---       |  ---        | --- 
icgc_sample_annotations_summary_table.txt | sample annotations used by PCAWG heterogeneity & evolution working group | [ICGC data portal](https://dcc.icgc.org/releases/PCAWG/evolution_and_heterogeneity)
PCAWG_sigProfiler_SBS_signatures_in_samples | counts of mutations attributed to each signature for PCAWG samples | [syn11738669.7](https://www.synapse.org/#!Synapse:syn11738669.7)
mutation_types_raw_counts.csv| mutation type counts in PCAWG samples | modified from [syn7357330](https://www.synapse.org/#!Synapse:syn7357330)
pcawg_cancer_types.csv | sample annotations used in [Jiao et. al](https://doi.org/10.1038/s41467-019-13825-8) | modified from [z-scores file](https://github.com/ICGC-TCGA-PanCancer/TumorType-WGS/blob/master/pcawg_mutations_types.csv)
pcawg_donor_clinical_August2016_v9.csv | donor clinical information in PCAWG | [ICGC data portal](https://dcc.icgc.org/releases/PCAWG/clinical_and_histology/)
pcawg_supplement_table1.csv | Supplementary Table 1. Sample, demographic and basic mutation data for the 2,583 white-listed donors in the PCAWG data-set | [PCAWG paper](https://doi.org/10.1038/s41586-020-1969-6)
pcawg-wgs-rnaseq-mirna.csv | Sample sheet for WGS | [ICGC data portal](https://dcc.icgc.org/releases/PCAWG/donors_and_biospecimens/)
clinical_ann_merged.csv | alloquot_id merged data | created with `merge_clinical_ann.py`, combines  icgc_sample_annotations_summary_table.txt, pcawg_supplement_table1.csv, pcawg_donor_clinical_August2016_v9.csv (in that priority order)


