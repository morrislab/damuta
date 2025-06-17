import wandb
import pickle
import numpy as np
import pandas as pd
import damuta as da

# infer activities only
run = wandb.init(project="damuta-paper", entity="harrig12", tags=['apobec-cellline'])

counts = pd.read_csv('data/petljack_2022_counts.csv', index_col=0)
counts = counts[da.constants.mut96]
annotation = pd.read_csv('data/apobec_ann.csv', index_col=0)
counts = counts[counts.index.isin(annotation.index)]
annotation = annotation.loc[counts.index]
counts = counts[~annotation['Non-Unique']]
annotation = annotation[~annotation['Non-Unique']]
#annotation = annotation.assign(**dict(zip(['Cell_Line', 'plate'], annotation.reset_index()['MutationType'].str.split('_', expand=True).values.T)))
#annotation['Experiment_constant']= 'constant'

data = da.DataSet(counts, annotation)
stoic_water = da.SignatureSet.from_damage_misrepair(pd.read_csv('~/damuta-paper/figure_data/h_phi.csv',index_col=0), pd.read_csv('~/damuta-paper/figure_data/h_eta.csv', index_col=0))


# Log model
wandb.config.setdefaults({
  "n_damage_sigs": 18,
  "n_misrepair_sigs": 6,
  "opt_method": "ADVI",
  "alpha_bias": 1, 
  "psi_bias": 0.1,
  "init_strategy": "kmeans",
  "seed": 3246,
  "type_col": "Experiment"
})

model = da.models.HierarchicalTandemLda(data, **wandb.config, phi_obs= stoic_water.damage_signatures.to_numpy(), 
                                        etaC_obs= stoic_water.misrepair_signatures.iloc[:,0:3].to_numpy(),
                                        etaT_obs= stoic_water.misrepair_signatures.iloc[:,3:6].to_numpy())

# Pick callbacks
cbs = [da.callbacks.LogELBO(every=100)]

# Fit model  
model.fit(n=10000, callbacks = cbs)

# Log metrics
hat = model.approx.sample(3)
LL = np.array(list(da.utils.mult_ll(data.counts, b) for b in hat.B)).sum(1).mean()

wandb.log({
  'LL': LL,
  'BIC': model.model.ndim * np.log(data.n_samples) - 2 * LL,
  'n_dim': model.model.ndim,
  'n_samples': data.n_samples, 
})
  
# Save model         
with open(f'results/ckpt/apobec', 'wb') as file: pickle.dump({"run_id": run.id, 'Model': model}, file)

# Clean up 
wandb.finish()



