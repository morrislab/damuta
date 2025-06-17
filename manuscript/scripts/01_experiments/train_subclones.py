import damuta as da
import pandas as pd
import numpy as np
import pickle
import wandb

run = wandb.init(project="damuta-paper", entity="harrig12", tags=['subclones'])

# subclone counts file is derived from pcawg restricted-access data and not included in the repo
counts = pd.read_csv('data/pcawg_subclone_counts.csv',  index_col=0, header=[0,1])

# Log data
counts = counts[da.constants.idx96]
counts.columns = da.constants.mut96
sel = counts.index.str.split('_S').str[0]
annotation = pd.read_csv('data/phg_clinical_ann.csv', index_col=0).loc[sel].reset_index()
annotation = annotation.set_index(counts.index)

data = da.DataSet(counts, annotation)
cosmic = da.SignatureSet(pd.read_csv('data/COSMIC_v3.2_SBS_GRCh37.txt', sep='\t', index_col = 0).T)

damuta_sigs = da.SignatureSet.from_damage_misrepair(pd.read_csv('results/figure_data/h_phi.csv',index_col=0), pd.read_csv('results/figure_data/h_eta.csv', index_col=0))

# Log model
wandb.config.setdefaults({
  "n_damage_sigs": 18,
  "n_misrepair_sigs": 6,
  "opt_method": "ADVI",
  "alpha_bias": 1, 
  "psi_bias": 0.1,
  "init_strategy": "kmeans",
  "seed": 3256,
  "type_col": "organ"
})

model = da.models.HierarchicalTandemLda(data, **wandb.config, phi_obs= damuta_sigs.damage_signatures.to_numpy(), 
                                        etaC_obs= damuta_sigs.misrepair_signatures.iloc[:,0:3].to_numpy(),
                                        etaT_obs= damuta_sigs.misrepair_signatures.iloc[:,3:6].to_numpy())
#model = da.models.TandemLda(data, **wandb.config)
#model = da.models.Lda(data, **wandb.config, init_signatures= cosmic)

# Pick callbacks
cbs = [da.callbacks.LogELBO(every=100)]

# Fit model  
model.fit(n=20000, callbacks = cbs)

# Log metrics
hat = model.approx.sample(20)
LL = np.array(list(da.utils.mult_ll(data.counts, b) for b in hat.B)).sum(1).mean()

wandb.log({
  'LL': LL,
  'BIC': model.model.ndim * np.log(data.n_samples) - 2 * LL,
  'ALP': model.ALP(),
  'LAP': model.LAP(), 
  'n_dim': model.model.ndim,
  'n_samples': data.n_samples, 
})
  
# Save model         
with open('results/ckpt/subclones', 'wb') as file: pickle.dump({"run_id": run.id, 'Model': model}, file)

# Clean up 
wandb.finish()
