import damuta as da
import pandas as pd
import numpy as np
import wandb

# Log data
counts = pd.read_csv('data/phg_counts.csv',  index_col=0)
annotation = pd.read_csv('data/phg_clinical_ann.csv',  index_col=0)

# Init run
data = da.DataSet(counts, annotation.loc[counts.index])
run = wandb.init(project="damuta-paper", tags=['stability'], entity="harrig12")

# Log model
wandb.config.setdefaults({
  "n_damage_sigs": 18,
  "n_misrepair_sigs": 6,
  "opt_method": "ADVI",
  "alpha_bias": 1, 
  "psi_bias": 0.1,
  "init_strategy": "kmeans",
  "type_col": "organ",
})

damuta = da.models.HierarchicalTandemLda(data, **wandb.config)
damuta._build_model(**damuta._model_kwargs)

wandb.log({
  'n_dim': damuta.model.ndim,
  'n_samples': data.n_samples, 
})

# Pick callbacks
cbs = [da.callbacks.LogELBO(every=100)]

# Fit model  
damuta.fit(n=20000, callbacks = cbs)

# Log metrics
hat = damuta.approx.sample(3)
LL = np.array(list(da.utils.mult_ll(data.counts, b) for b in hat.B)).sum(1).mean()

wandb.log({
  'LL': LL,
  'BIC': damuta.model.ndim * np.log(data.n_samples) - 2 * LL,
})
  
# Save model         
#with open(f'ckpt/{run.id}', 'wb') as file: pickle.dump({"run_id": run.id, 'Model': damuta}, file)

# Save sigs for stability check
pd.DataFrame(np.vstack(hat.phi), columns = da.mut32).to_csv(f'results/figure_data/stability/{run.id}_phi.csv')
pd.DataFrame(np.vstack(hat.eta).reshape(-1,6), columns = da.mut6).to_csv(f'results/figure_data/stability/{run.id}_eta.csv')

# Clean up 
wandb.finish()

