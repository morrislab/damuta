import numpy as np
import pymc3 as pm
from config import *
import theano.tensor as tt 
import typing
# Constants
C = 32
M = 3

@extyaml
def collapsed_model_factory(corpus, J: int, K: int, alpha_bias: float, psi_bias: float, 
                            gamma_bias: float, beta_bias: float, phi_obs = None, eta_obs = None,):
    
    S = corpus.shape[0]
    N = corpus.sum(1).reshape(S,1)
    etaC_obs = eta_obs[0:16] if eta_obs is not None else eta_obs
    etaT_obs = eta_obs[16:32] if eta_obs is not None else eta_obs
    
    with pm.Model() as model:
        
        data = pm.Data("data", corpus)
        phi = pm.Dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C), observed=phi_obs)
        theta = pm.Dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = pm.Dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones((K,4)) * beta_bias
        etaC = pm.Dirichlet("etaC", a=beta[:,[0,2,3]], shape=(C//2, K, M), observed=etaC_obs)
        etaT = pm.Dirichlet("etaT", a=beta[:,[0,1,2]], shape=(C//2, K, M), observed=etaT_obs)
        eta = pm.Deterministic('eta', pm.math.concatenate([etaC, etaT], axis=0))
        
        B = pm.Deterministic("B", (pm.math.matrix_dot(theta, phi)[:,:,None] * \
                                   pm.math.matrix_dot(tt.batched_dot(theta,A),eta)).reshape((S, -1)))
    
        # mutation counts
        lik = pm.Multinomial('corpus', n = N, p = B , observed=data)
        

    return model

def get_tau(phi, eta):
    assert len(phi.shape) == 2 and len(eta.shape) == 3
    J,C = phi.shape
    C,K,M = eta.shape 
    
    tau = np.einsum('jc,ckm->jkcm',phi, eta).reshape(-1,96)
    
    #sel = np.concatenate([np.arange(0,46,3), np.arange(1,47,3), np.arange(2,48,3),
    #            np.arange(48,94,3), np.arange(49,95,3), np.arange(50,96,3)])
    
    # reorder by block
    #tau = np.vstack([t[sel] for t in tau])
    return tau

def get_phis(sigs):
    # for each signature in sigs, get the corresponding phi
    wrapped = sigs.reshape(sigs.shape[0], -1, 16)
    phis = np.hstack([wrapped[:,[0,1,2],:].sum(1), wrapped[:,[3,4,5],:].sum(1)])
    return phis

def get_etas(sigs):
    # for each signature in sigs, get the corresponding eta
    # ordered alphabetically, or pyrimidine last (?)
    wrapped = sigs.reshape(sigs.shape[0], -1, 16)
    etas = wrapped.sum(2)
    return etas#[0,1,2,3,5,4]
