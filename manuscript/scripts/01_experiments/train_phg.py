import damuta as da
import pandas as pd
import pymc as pm
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
  "likelihood": "sequential",
  "seed": 3242,
  "type_col": "organ",
})

model = da.models.HierarchicalTandemLda(data, **wandb.config)
#model = da.models.Lda(data, **wandb.config)

model._build_model(**model._model_kwargs)

ndim = sum(v.size for v in model.model.initial_point().values())

wandb.log({
  'n_dim': ndim,
  'n_samples': data.n_samples, 
})

# Pick callbacks
stop = da.callbacks.StopOnConvergence(n_mutations=data.counts.to_numpy().sum())
cbs = [da.callbacks.LogELBO(every=100), stop]

# Fit model  
model.fit(
    n=50000, callbacks=cbs, obj_optimizer=pm.adam(learning_rate=0.02),
    backend="jax"
)

# Log metrics
hat = model._sample(3)
LL = np.array(list(da.utils.mult_ll(data.counts, b) for b in hat.B)).sum(1).mean()

wandb.log({
  'converged_step': stop.converged_step,
  'LL': LL,
  'BIC': ndim * np.log(data.n_samples) - 2 * LL,
})
  
# Save model         
with open(f'results/ckpt/phg', 'wb') as file: pickle.dump({"run_id": run.id, 'Model': model}, file)

# Clean up 
wandb.finish()
