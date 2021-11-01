# script to create pcawg_clinical_ann.csv and hartwig_clinical_ann.csv
import pandas as pd

## pcawg
donor_clinical = pd.read_csv('pcawg_donor_clinical_August2016_v9.csv')
icgc_annotations = pd.read_csv('icgc_sample_annotations_summary_table.txt', sep = '\t').set_index('tumour_aliquot_id')
pcawg_annotations = pd.read_csv('pcawg_supplement_table1.csv').set_index('tumour_specimen_aliquot_id')
dt_types = pd.read_csv('pcawg_cancer_types.csv', index_col=0)

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
#clinical_ann.to_csv('clinical_ann_merged.csv')

clinical_ann = clinical_ann[[
    'pcawg_class',
    'tumour_type',
    'project_code',
    'reported_sex',
    'donor_survival_time',
    'donor_age_at_diagnosis',
    'tumour_stage',
    'tumour_grade',
    'first_therapy_type',
    'first_therapy_response',
    'specimen_donor_treatment_type',
    'histology_tier1',
    'histology_tier2',
    'histology_tier3',
    'histology_tier4',
    'tumour_histological_type',
    'ancestry_primary',
    'donor_vital_status',
    'donor_interval_of_last_followup',
    'tobacco_smoking_history_indicator',
    'tobacco_smoking_intensity',
    'alcohol_history',
    'alcohol_history_intensity'
]]

clinical_ann.to_csv('pcawg_clinical_ann.csv')

## hartwig
hw = pd.read_csv('sample.cancertype.mapping.csv', index_col=0)
hw = hw[['pcawgClass','primaryTumorLocation', 'cancerSubtype', 'tumourPurity']]
hw = hw.rename(columns = {'pcawgClass': 'pcawg_class',
                    'primaryTumorLocation': 'primary_tumour_location', 
                    'cancerSubtype': 'cancer_subtype', 
                    'tumourPurity': 'tumour_purity'})
hw.to_csv('hartwig_clinical_ann.csv')
