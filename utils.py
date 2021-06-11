import numpy as np
import pymc3 as pm
from config import *
from plotting import *
import theano.tensor as tt 
import typing

def collapsed_model_factory(corpus, J: int, K: int, alpha_bias: float, psi_bias: float, 
                            gamma_bias: float, beta_bias: float, phi_obs = None, eta_obs = None,):
    
    S = corpus.shape[0]
    N = corpus.sum(1).reshape(S,1)
    etaC_obs = eta_obs[0,None,:,:] if eta_obs is not None else eta_obs
    etaT_obs = eta_obs[1,None,:,:] if eta_obs is not None else eta_obs
    
    with pm.Model() as model:
        
        data = pm.Data("data", corpus)
        phi = pm.Dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C), observed=phi_obs)
        theta = pm.Dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = pm.Dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones((K,4)) * beta_bias
        etaC = pm.Dirichlet("etaC", a=beta[:,[0,2,3]], shape=(1, K, M), observed=etaC_obs)
        etaT = pm.Dirichlet("etaT", a=beta[:,[0,1,2]], shape=(1, K, M), observed=etaT_obs)
        eta = pm.Deterministic('eta', pm.math.concatenate([etaC, etaT], axis=0))

        B = pm.Deterministic("B", (pm.math.matrix_dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                   pm.math.matrix_dot(tt.batched_dot(theta,A),eta)[:,:,:,None]).reshape((S, -1)))
        
        # mutation counts
        lik = pm.Multinomial('corpus', n = N, p = B , observed=data)

    return model

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

def get_tau(phi, eta):
    assert len(phi.shape) == 2 and len(eta.shape) == 3
    tau =  np.einsum('jpc,pkm->jkpmc', phi.reshape((-1,2,16)), eta).reshape((-1,96))
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

def flatten_eta(eta): # eta pkm -> kc
    return np.moveaxis(eta,0,1).reshape(-1, 6)
        
