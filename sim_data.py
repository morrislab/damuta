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
    C = 32
    M = 3
    J = 2
    K = 2
    N = np.array([1000] * 10)
    S = len(N)
    
    # Hyper-parameter for priors
    alpha = np.ones(C) * 0.05
    #alpha[0] = 1
    psi = np.ones(J)
    gamma = np.ones(K) * 0.1
    #gamma = np.array([5,0.1,0.1,0.1]) * 0.1
    beta = np.ones((K,4)) * 0.1
    #beta = np.repeat(np.array([[1, 0.1, 1, 5]]), K, axis=0)
    
    phi_gen = pm.Dirichlet.dist(a=alpha, shape=(C)).random(size = J)
    theta_gen = pm.Dirichlet.dist(a=psi).random(size = S)
    A_gen = pm.Dirichlet.dist(a=gamma, shape=(J, K)).random(size = S)
    # ACGT
    # 0123
    eta_gen = np.vstack([[pm.Dirichlet.dist(a=beta[:,[0,2,3]]).random(size=1)] * 16, 
                         [pm.Dirichlet.dist(a=beta[:,[0,1,2]]).random(size=1)] * 16]).squeeze()
    #eta_gen = np.vstack([pm.Dirichlet.dist(a=beta[:,[0,2,3]]).random(size=C//2), 
    #                     pm.Dirichlet.dist(a=beta[:,[0,1,2]]).random(size=C//2)])
    
    W=(theta_gen@phi_gen).T
    Q=np.einsum('sj,sjk->sk', theta_gen, A_gen)@eta_gen
    B=np.einsum('cs,csm->scm',W,Q).reshape(S, -1)
    data = np.vstack([d.random(size = 1) for d in map(pm.Multinomial.dist, N, B)])
    

def encode_counts(counts):
    # turn counts of position into word-style encoding
    # https://laptrinhx.com/topic-modeling-with-pymc3-398251916/
    # A[C>A]A, A[C>G]A, A[C>T]A, A[C>A]C... T[T>G]T
    
    x32 = np.tile([np.arange(16), np.arange(16, 32)], 3).reshape(-1)
    y6 = np.repeat([0,1,2,0,2,1], 16)
    
    S, C = counts.shape
    sel = [np.repeat(range(C), counts[i].astype(int)) for i in range(S)]

    X = [x32[s] for s in sel]
    Y = [y6[s] for s in sel]
    
    return X, Y