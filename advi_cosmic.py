# advi_cosmic.py

import numpy as np
import pymc3 as pm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import typing
import logging
import theano
from plotting import *
from sim_data import sim_cosmic
import config
import theano.tensor as tt 

# Constants
C = 32
M = 3

@config.extyaml
def fit_collapsed_model(corpus_obs: np.ndarray, J: int, K: int,
                        alpha_bias: float, psi_bias: float, gamma_bias: float, beta_bias: float, 
                        n_steps: int, seed: int, lr: float, init_method: str) -> (pm.model.Model, pm.variational.inference.ADVI):
    
    logging.debug(f"theano device: {theano.config.device}")
    
    S = corpus_obs.shape[0]
    N = corpus_obs.sum(1)
    
    logging.debug(f"number of samples in corpus: {S}")
    logging.debug(f"mean number of mutations per sample: {N.mean()}")
    
    with pm.Model() as model:
    
        phi = pm.Dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C))
        theta = pm.Dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = pm.Dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones((K,4)) * beta_bias
        etaC = pm.Dirichlet("etaC", a=beta[:,[0,2,3]], shape=(C//2, K, M))
        etaT = pm.Dirichlet("etaT", a=beta[:,[0,1,2]], shape=(C//2, K, M))
        eta = pm.Deterministic('eta', pm.math.concatenate([etaC, etaT], axis=0))
    
        B = pm.Deterministic("B", (pm.math.matrix_dot(theta, phi)[:,:,None] * \
                                   pm.math.matrix_dot(tt.batched_dot(theta,A),eta)).reshape((S, -1)))
        
        # mutation counts
        pm.Multinomial('corpus', n = N.reshape(S,1), p = B , observed=corpus_obs)
    
    with model:
        trace = pm.ADVI()
        trace.fit(n_steps)
    
    return model, trace

class rv_init():
    
    def __init__(self, method='uniform', **kwargs):
        
        self.phi = None
        self.eta = None
        
        if method == 'uniform':
                return self
        elif method == 'from-tau':
                return split_tau(sigs, J, K)
        elif method == 'clark':
                return clark(data, J, K) 
            
    
    def split_tau(self, sigs=None, J=1, K=1):
        assert sigs is not None
        # for each signature in sigs, get the corresponding phi
        wrapped = sigs.reshape(sigs.shape[0], -1, 16)
        self.phi = np.hstack([wrapped[:,[0,1,2],:].sum(1), wrapped[:,[3,4,5],:].sum(1)])[range(J)]

        # for each signature in sigs, get the corresponding eta
        # TODO: ordered alphabetically, or pyrimidine last (?)
        wrapped = sigs.reshape(sigs.shape[0], -1, 16)
        self.eta = wrapped.sum(2)
          
        # subset to desired J & K
        self.phi = self.phi[range(J)]
        self.eta = self.eta[range(K)]
        
        return self
    
    def clark(self, data=None, J=1, K=1):
        assert data is not None
               


def main():
    cosmic, tau, tau_activities =sim_cosmic()
    with PdfPages('true_sigs.pdf') as pdf:
        for i in range(tau.shape[0]):
            plot_tau(tau[i])
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    model, trace = fit_collapsed_model(corpus_obs = cosmic)
    plot_diagnostics(trace, 3, 2)
    
    
    
if __name__ == '__main__':
    main()
