# script to create clinical_ann_filtered.csv
import pandas as pd

# filter annotations for useful info
clinical_ann = pd.read_csv('data/clinical_ann_merged.csv', index_col = 0)
clinical_ann = clinical_ann[[
    'type',
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

clinical_ann.to_csv('data/clinical_ann_filtered.csv')