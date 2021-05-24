# advi_cosmic.py

import numpy as np
import pymc3 as pm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import typing
import logging
import theano
from config import *
from utils import *
from plotting import *
from sim_data import sim_cosmic

import wandb



@extyaml
def fit_collapsed_model(corpus_obs: np.ndarray, callbacks, 
                        n_steps: int, seed: int, lr: float) -> (pm.model.Model, pm.variational.inference.ADVI):
    
    logging.debug(f"theano device: {theano.config.device}")
    
    S = corpus_obs.shape[0]
    N = corpus_obs.sum(1)
    
    logging.debug(f"number of samples in corpus: {S}")
    logging.debug(f"mean number of mutations per sample: {N.mean()}")
    
    with collapsed_model_factory(corpus_obs) as model:
        
        wandb.log({'graphical model': wandb.Image(save_gv(model))})

        trace = pm.ADVI()
        trace.fit(n_steps, callbacks = callbacks)
   
    return model, trace


@extyaml  
def cbs(*args, train=None, val=None, log_every=None, tau_gt=None):
    # return a list of callbacks, with extra parameters as desired
    
    def wandb_calls(*args):
        approx, losses, i = args
        wandb.log({'ELBO': losses[-1]})
        
        # only log expensive objects (plots) sparcely
        if i % log_every == 1 :
            
            hat = approx.sample(1000)
            tau_hat = get_tau(hat.phi.mean(0), hat.eta.mean(0))
            
            
            wandb.log({         
                       'train alp': alp_B(train, hat.B.mean(0)),
                       'val alp': alp_B(val, hat.B.mean(0)),
                       'phi posterior': plot_phi_posterior(hat.phi, cols=None),
                       'eta posterior': plot_eta_posterior(hat.eta, cols=None),
                       'signature similarities': plot_tau_cos(tau_gt, hat.phi.mean(0), hat.eta.mean(0)),
                        
                       'inferred phi': plot_phi(hat.phi.mean(0)),
                       'inferred eta': plot_eta(hat.eta.mean(0)),
                       'inferred signatures': plot_tau(tau_hat),
                        
                       '1000 samples from phi node': plot_mean_std(hat.phi),
                       '1000 samples from eta node': plot_mean_std(hat.eta),
                       '1000 samples from theta node': plot_mean_std(hat.theta),
                       '1000 samples from A node': plot_mean_std(hat.A),
                       '1000 samples from B node': plot_mean_std(hat.B),

                      })
    
    return [wandb_calls]

def alp_B(data, B):
    return (data * np.log(B)).sum() / data.sum()

def split_count(counts, fraction):
    c = (counts * fraction).astype(int)
    frac1 = np.histogram(np.repeat(np.arange(96), c), bins=96, range=(0, 96))[0]
    frac2 = counts - frac1
    assert all(frac2 >= 0) and all(frac1 >= 0)
    return frac1, frac2

def split_by_count(data, fraction=0.8):
    stacked = np.array([split_count(m, fraction) for m in data])
    return stacked[:,0,:], stacked[:,1,:]

def split_by_S(data, fraction=0.8):
    c = int((data.shape[0] * fraction))
    frac1 = data[0:c]
    frac2 = data[c:(data.shape[0])]
    return frac1, frac2

def main():
    wandb.init()
    wandb.config.update(config)
    
    cosmic, tau, tau_activities = sim_cosmic()
    wandb.log({'gt signatures': plot_tau(tau).update_layout(height = 800)})
    
    train, val  = split_by_count(cosmic)

    model, trace = fit_collapsed_model(corpus_obs = train, 
                                       callbacks = cbs(train = train, val = val, 
                                                       log_every=100, tau_gt = tau))
    
    
    
    
    
    
    
    

    
    
    
if __name__ == '__main__':
    main()
