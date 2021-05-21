# sim_data.py
# simulate mutation catalogue using cosmic signatures

import numpy as np
import pymc3 as pm
import pandas as pd
import theano
import theano.tensor as tt 
import typing
import logging
from config import extyaml

@extyaml
def sim_cosmic(sig_defs: str, S: int, N: int, tau_hyperprior: float, I: int, seed: int):
    # simulate from a predefined set of signatures (cosmic format)
    np.random.seed(seed)
    pm.set_tt_rng(seed)
    N = np.array([N] * S)
    tau = pd.read_csv(sig_defs, index_col = [0,1]).to_numpy().T

    if I:
        tau = tau[np.random.choice(tau.shape[0], size = I, replace = False)]                                       
    # draw activities according to tau
    tau_activities = pm.Dirichlet.dist(a=np.ones(I) * tau_hyperprior).random(size = S)
    B=(tau_activities @ tau)
    logging.debug(f'generated tau_activities shape {tau_activities.shape}')
    logging.debug(f'generated sigs (cosmic) shape {tau.shape}')
    corpus = np.vstack([d.random(size = 1) for d in map(pm.Multinomial.dist, N, B)])
    logging.debug(f'generated corpus shape {corpus.shape}')
    return corpus, tau, tau_activities

@extyaml
def sim_parametric():
    return None



