import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics  import accuracy_score, auc, roc_curve, precision_recall_curve, roc_auc_score, precision_score, recall_score, average_precision_score, f1_score, matthews_corrcoef
from lightgbm import LGBMClassifier

fp = f'figures'
os.makedirs(fp, exist_ok=True)

# set up variables to predict
ann = pd.read_csv('figure_data/phg_clinical_ann.csv', index_col=0, low_memory = False)
maf  = pd.read_csv("~/pcawg_DDR_alts/DNA_metabolism_nonsyn.maf", sep='\t')
maf = maf[maf['Variant_Type'] =="SNP"]
maf = maf[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Chromosome', 'Start_position', 'Variant_Classification', 'Reference_Allele', 'Tumor_Seq_Allele2']]
pathway_membership = pd.read_csv('figure_data/ddr_dnam.csv').rename(columns =  {'gene':'Hugo_Symbol', 'gui':'Tumor_Sample_Barcode'})

# subset to just pcawg
sel = (ann.cohort == 'PCAWG')
gene = pd.get_dummies(maf, columns = ['Hugo_Symbol']).groupby('Tumor_Sample_Barcode').sum().filter(like = 'Hugo_Symbol_')
pthw = pd.get_dummies(pd.merge(maf, pathway_membership), columns = ['pathway']).groupby('Tumor_Sample_Barcode').sum().filter(like = 'pathway_')
pthw = pthw.reindex(ann.index[sel], fill_value= 0)
gene = gene.reindex(ann.index[sel], fill_value= 0)
ann = pd.concat([ann[sel],pthw,gene], axis = 1)

W = pd.read_csv('figure_data/h_W.csv', index_col = 0).loc[ann.index].to_numpy().reshape(-1,18,6)
gamma = pd.DataFrame(W.sum(1), index = ann.index)
theta = pd.DataFrame(W.sum(2), index = ann.index)
gamma = gamma.rename(columns=lambda x: 'M' + str(x+1))
theta = theta.rename(columns=lambda x: 'D' + str(x+1))

deconstructsigs_activities = pd.read_csv('figure_data/phg_deconstructsigs_activities.csv', sep = '\t', index_col=0).drop_duplicates().reindex(ann.index)
deconstructsigs_activities = deconstructsigs_activities.loc[:,deconstructsigs_activities.sum(0) > 0]

def evaluate_classifier(model, X_test, y_test):
    predictions = model.predict_proba(X_test) 
    accuracy  = accuracy_score(y_test, predictions[:,1] >= 0.5)
    roc_auc   = roc_auc_score(y_test, predictions[:,1])
    precisions, recalls, thresholds = precision_recall_curve(y_test, predictions[:,1])
    pr_auc = auc(recalls, precisions)
    precision = precision_score(y_test, predictions[:,1]>=0.5)
    recall    = recall_score(y_test, predictions[:,1]>=0.5)
    ap    = average_precision_score(y_test, predictions[:,1])
    mcc = matthews_corrcoef(y_test, predictions[:,1] >= 0.5)
    f1 = f1_score(y_test, predictions[:,1] >= 0.5)
    baseline_ap = (y_test.sum())/y_test.shape[0]
    baseline_acc = (y_test.shape[0]-y_test.sum())/y_test.shape[0]
    result = pd.DataFrame(
        [[ap, accuracy, precision, recall, roc_auc, pr_auc, mcc, f1, baseline_ap, baseline_acc]],
        columns=['AP', 'Accuracy', 'Precision', 'Recall', 'ROC_auc', 'PR_auc', 'MCC', 'F1', 'baseline_ap', 'baseline_acc']
    )
    return(result)

def run_experiment(activities, y, model_class=LGBMClassifier, n = 10, **kwargs):
    models = {}
    metrics = {}
    test_sets = {}
    for i in range(n):
        # Compose dataset
        X_train, X_test, y_train, y_test = train_test_split(activities, y, test_size = 0.3, stratify=y, random_state = i)
        # Train Model
        models[i] = model_class(**kwargs)
        models[i].fit(X_train, y_train)
        # Evaluate results
        metrics[i] = evaluate_classifier(models[i], X_test, y_test)
        test_sets[i] = X_test.index.to_list()
    return({'metrics':pd.concat(metrics), 'models':models, 'test_sets':test_sets})

def train_tasks(X, ys, n=10):
    # train a classifier for each pathway
    models = {}
    test_sets = {}
    metrics = pd.DataFrame()
    for task in ys.columns:
        print(task)
        sel = (~ys[task].isnull())
        activities=X.loc[sel]
        y = (ys[task].loc[sel]>0).astype(int)
        exp = run_experiment(activities, y, n=n, n_estimators=30, n_jobs=n)
        exp['metrics']['task'] = task
        metrics = pd.concat([metrics, exp['metrics']])
        models.update({task: exp['models']})
        test_sets.update({task: exp['test_sets']})
    return({'metrics': metrics, 'models': models, 'test_sets': test_sets})

#########################################################
# lightGBM
#########################################################
# binary tree classification
n_seeds = 10
tasks = ann.filter(like='pathway_').columns.to_list()

model_0_2 = train_tasks(deconstructsigs_activities * ann['n_mut'][:,None], ann[tasks]>0, n=n_seeds)
model_0_2['name'] = 'model_0_2'

_t = theta.copy()
_g = gamma.copy()
_t[_t <=0.05] = 0
_g[_g <=0.05] = 0

model_1_3 = train_tasks(pd.concat([_t*ann['n_mut'][:,None], _g*ann['n_mut'][:,None]], axis=1), ann[tasks]>0, n=n_seeds)
model_1_3['name'] = 'model_1_3'

model_1_4 = train_tasks(_g*ann['n_mut'][:,None], ann[tasks]>0, n=n_seeds)
model_1_4['name'] = 'model_1_4'

model_1_5 = train_tasks(_t*ann['n_mut'][:,None], ann[tasks]>0, n=n_seeds)
model_1_5['name'] = 'model_1_5'

model_5_0 = train_tasks(pd.DataFrame(ann['n_mut']), ann[tasks]>0, n=n_seeds)
model_5_0['name'] = 'model_5_0'

#########################################################
# mutated sample classification
#########################################################

class_balance = ann[tasks]>1
class_balance.to_csv(f'{fp}/class_balance.csv')

full_classifier_metrics = pd.concat(m['metrics'].assign(model=m['name']) for m in [model_0_2,model_1_3,model_1_4,model_5_0])
# subset to metrics of interest
full_classifier_metrics = full_classifier_metrics[['Accuracy','Precision', 'Recall', 'ROC_auc', 'PR_auc', 'AP', 'MCC', 'F1', 'task','baseline_ap','baseline_acc', 'model']]
# add seed column
full_classifier_metrics['seed'] = np.tile(np.arange(1,n_seeds+1), len(full_classifier_metrics)//n_seeds)
full_classifier_metrics.to_csv(f'{fp}/full_classifier_metrics.csv', index=False)

print(full_classifier_metrics.groupby(['model', 'task']).mean().round(2))

#########################################################
# feature importance
#########################################################

damuta_imp = pd.concat([pd.DataFrame({
    'feature' : model_1_3['models'][task][seed].feature_name_, 
    'split_importance' : model_1_3['models'][task][seed].feature_importances_,
    'pathway': task, 'seed': seed,
    'model': 'model_1_3'}) for task in tasks for seed in range(n_seeds)])

misrepair_imp = pd.concat([pd.DataFrame({
    'feature' : model_1_4['models'][task][seed].feature_name_, 
    'split_importance' : model_1_4['models'][task][seed].feature_importances_,
    'pathway': task, 'seed': seed,
    'model': 'model_1_4'}) for task in tasks for seed in range(n_seeds)])

damage_imp = pd.concat([pd.DataFrame({
    'feature' : model_1_5['models'][task][seed].feature_name_, 
    'split_importance' : model_1_5['models'][task][seed].feature_importances_,
    'pathway': task, 'seed': seed,
    'model': 'model_1_5'}) for task in tasks for seed in range(n_seeds)])

cosmic_imp = pd.concat([pd.DataFrame({
    'feature' : model_0_2['models'][task][seed].feature_name_, 
    'split_importance' : model_0_2['models'][task][seed].feature_importances_,
    'pathway': task, 'seed': seed,
    'model': 'model_0_2'}) for task in tasks for seed in range(n_seeds)])

tmb_imp = pd.concat([pd.DataFrame({
    'feature' : model_5_0['models'][task][seed].feature_name_, 
    'split_importance' : model_5_0['models'][task][seed].feature_importances_,
    'pathway': task, 'seed': seed,
    'model': 'model_5_0'}) for task in tasks for seed in range(n_seeds)])

pd.concat([damuta_imp, misrepair_imp, damage_imp, cosmic_imp, tmb_imp]).to_csv(f'{fp}/feature_importances.csv')





