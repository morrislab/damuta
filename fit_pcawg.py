# fit_pcawg.py

import numpy as np
import pymc3 as pm
import pandas as pd
import theano
from config import *
from utils import *
from plotting import *

import wandb

@extyaml
def fit_collapsed_model(corpus_obs: np.ndarray, J: int, K: int,
                        alpha_bias: float, psi_bias: float, 
                        gamma_bias: float, beta_bias: float,
                        callbacks: list, n_steps: int, seed: int) -> (pm.model.Model, pm.variational.inference.ADVI):
    
    logging.debug(f"theano device: {theano.config.device}")
    
    S = corpus_obs.shape[0]
    N = corpus_obs.sum(1)
    
    logging.debug(f"number of samples in corpus: {S}")
    logging.debug(f"mean number of mutations per sample: {N.mean()}")
    
    with collapsed_model_factory(corpus_obs, J, K, alpha_bias, 
                                 psi_bias, gamma_bias, beta_bias) as model:
        
        wandb.log({'graphical model': wandb.Image(save_gv(model))})

        trace = pm.ADVI()
        trace.fit(n_steps, callbacks = callbacks)
   
    return model, trace


@extyaml  
def cbs(*args, train=None, val=None, tau_gt=None, log_every=None):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
        # only log expensive objects (plots) sparcely
        if i % log_every == 1 :
            
            hat = approx.sample(1000)
            tau_hat = get_tau(hat.phi.mean(0), hat.eta.mean(0))
            act_hat = np.einsum('sj,sjk->sjk', hat.theta.mean(0), hat.A.mean(0))
            
            wandb.log({         
                       'train alp': alp_B(train, hat.B.mean(0)),
                       'val alp': alp_B(val, hat.B.mean(0)),
                       'phi posterior': plot_phi_posterior(hat.phi, cols=None),
                       'eta posterior': plot_eta_posterior(hat.eta, cols=None),
                       #'signature similarities': plot_cossim(tau_gt, tau_hat),
                       'phi clustering': go.Figure(go.Heatmap(z=cosine_similarity(hat.phi.mean(0), hat.phi.mean(0)).round(2), colorscale = 'viridis')),
                       'eta clustering': go.Figure(go.Heatmap(z=cosine_similarity(flatten_eta(hat.eta.mean(0)),flatten_eta(hat.eta.mean(0))).round(2), colorscale = 'viridis')),
                        
                       'inferred signatures': plot_tau(tau_hat),
                       'inferred phi': plot_phi(hat.phi.mean(0)),
                       'inferred eta': plot_eta(hat.eta.mean(0)),

                       'estimated recruitment': plot_mean_std(act_hat),
                        
                       'average availability': plot_bipartite_J(act_hat.mean(0)),
                       'average recruitment': plot_bipartite_K(act_hat.mean(0)),
                       'availability in sample 0': plot_bipartite_J(act_hat[0]),
                       'recruitment in sample 0': plot_bipartite_K(act_hat[0]),

                      })
    
    return [wandb_calls]

@extyaml
def load_counts(counts_fn, types_fn, type_subset):

    counts = pd.read_csv(counts_fn, index_col = [0], header = [0])[mut96]
    cancer_types = pd.read_csv(types_fn)
    cancer_types.columns=['type', 'guid']

    if type_subset is not None:
        counts = counts.loc[cancer_types.guid[[x in type_subset for x in cancer_types.type]]]

    return counts.to_numpy()
    

def main():
    wandb.init(project = 'da-pcawg',
               tags=['local signatures'] + config['fit_pcawg.py']['load_counts']['type_subset'])
    wandb.config.update(config)
    
    counts = load_counts()
    train, val  = split_by_count(counts)

    model, trace = fit_collapsed_model(corpus_obs = train, 
                                       callbacks = cbs(train = train, val = val))
    
    
if __name__ == '__main__':
    main()