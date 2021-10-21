# script to create clinical_ann_merged.csv
import pandas as pd

donor_clinical = pd.read_csv('data/pcawg_donor_clinical_August2016_v9.csv')
icgc_annotations = pd.read_csv('data/icgc_sample_annotations_summary_table.txt', sep = '\t').set_index('tumour_aliquot_id')
pcawg_annotations = pd.read_csv('data/pcawg_supplement_table1.csv').set_index('tumour_specimen_aliquot_id')
dt_types = pd.read_csv('data/pcawg_cancer_types.csv', index_col=0)

pcawg_annotations.update(icgc_annotations)
icgc_annotations.update(pcawg_annotations)
clinical_ann = icgc_annotations.merge(pcawg_annotations, how = 'left', 
                                  left_index = True, right_index = True,
                                  on= ['histology_abbreviation',
                                       'icgc_sample_id',
                                       'icgc_donor_id',
                                       'tumour_stage',
                                       'tumour_grade',
                                       'specimen_donor_treatment_type'
                                       ])
clinical_ann = clinical_ann.reset_index()
clinical_ann = clinical_ann.merge(donor_clinical, how = 'left', on  = ['icgc_donor_id', 
                                                                       'project_code', 
                                                                       'donor_wgs_included_excluded',    
                                                                       'donor_unique_id', 
                                                                       'submitted_donor_id', 
                                                                       'tcga_donor_uuid',
                                                                       'donor_survival_time',
                                                                       'donor_age_at_diagnosis',
                                                                       'first_therapy_type',
                                                                       'first_therapy_response',
                                                                       
                                                                      ])

clinical_ann = clinical_ann.set_index('tumour_aliquot_id')
clinical_ann = dt_types.merge(clinical_ann, right_index = True, left_index = True)
clinical_ann.to_csv('data/clinical_ann_merged.csv')