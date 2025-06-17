import damuta as da
import pandas as pd
import numpy as np
import pickle
import wandb


run = wandb.init(project="damuta-paper", entity="harrig12", tags=['pcawg-hmf-gel'])

# Log data
counts = pd.read_csv('data/phg_counts.csv', index_col=0)
annotation = pd.read_csv('data/phg_clinical_ann.csv',  index_col=0)
data = da.DataSet(counts, annotation)

# Log model
wandb.config.setdefaults({
  "n_damage_sigs": 18,
  "n_misrepair_sigs": 6,
  "opt_method": "ADVI",
  "alpha_bias": 1, 
  "psi_bias": 0.1,
  "init_strategy": "kmeans",
  "seed": 3242,
  "type_col": "organ",
})

model = da.models.HierarchicalTandemLda(data, **wandb.config)
#model = da.models.Lda(data, **wandb.config)

model._build_model(**model._model_kwargs)
model.model.ndim

wandb.log({
  'n_dim': model.model.ndim,
  'n_samples': data.n_samples, 
})

# Pick callbacks
cbs = [da.callbacks.LogELBO(every=100)]

# Fit model  
model.fit(n=20000, callbacks = cbs)

# Log metrics
hat = model.approx.sample(3)
LL = np.array(list(da.utils.mult_ll(data.counts, b) for b in hat.B)).sum(1).mean()

wandb.log({
  'LL': LL,
  'BIC': model.model.ndim * np.log(data.n_samples) - 2 * LL,
})
  
# Save model         
with open(f'results/ckpt/phg', 'wb') as file: pickle.dump({"run_id": run.id, 'Model': model}, file)

# Clean up 
wandb.finish()
