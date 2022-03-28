import numpy as np
from pandas.core.frame import DataFrame
from sklearn.cluster import k_means
from .utils import get_phi, get_eta
import warnings

def init_sigs(strategy, data=None, J=None, K=None, tau=None, seed=42):
    warnings.warn("init_sigs is deprecated, see Damuta class", DeprecationWarning)
    rng=np.random.default_rng(seed)
    
    strats = ['kmeans', 'supply_tau', 'uniform', 'random']
    assert strategy in strats, f'strategy should be one of {strats}'
    
    if strategy == 'kmeans':
        phi, eta = init_kmeans(data, J, K, rng) 
    elif strategy == 'supply_tau':
        phi, eta = init_from_tau(tau, J, K, rng)
    elif strategy == 'random':
        phi, eta = init_random(J, K, rng)
    elif strategy == 'uniform':
        # default from pymc3
        phi, eta = None, None
    
    # eta should be kxpxm
    if eta is not None:
        etaC, etaT = (eta[:,0,:], eta[:,1,:])
        assert np.allclose(etaC.sum(1), 1) and np.allclose(etaT.sum(1), 1)
    else: etaC, etaT = (None, None)
    
    return phi, etaC, etaT
    
def init_kmeans(data, J, K, rng):
    warnings.warn("init_kmeans is deprecated, see Damuta class", DeprecationWarning)
    if isinstance(data, DataFrame):
        data = data.to_numpy()
    
    # get proportions for signature initialization
    data = data/data.sum(1)[:,None]
    
    #return kmeans_alr(get_phi(data), J, rng), kmeans_alr(get_eta(data).reshape(-1,P*M), K, rng).reshape(-1,P,M) 
    return k_means(get_phi(data), J, init='k-means++',random_state=np.random.RandomState(rng.bit_generator))[0], \
           k_means(get_eta(data).reshape(-1,6), K, init='k-means++', random_state=np.random.RandomState(rng.bit_generator))[0].reshape(-1,P,M)
 
def init_from_tau(tau, J, K, rng):
    warnings.warn("init_from_tau is deprecated, see Damuta class", DeprecationWarning)
    # return phi and eta naively from tau
    # if I > J/K , I is randomly sampled
    # if I < J/K , taus will be  sampled with replacement
    
    if isinstance(tau, DataFrame):
        tau = tau.to_numpy()
    
    assert np.allclose(tau.sum(1), 1)
    
    I = tau.shape[0]
    phi = rng.choice(get_phi(tau), size = J, replace = I>J)
    eta = rng.choice(get_eta(tau), size = K, replace = I>K)
    
    return phi, eta
   
def init_random(J, K, rng, sparsity = 0.1):
    warnings.warn("init_random is deprecated, see Damuta class", DeprecationWarning)
    # a random, but non-uniform initialization
    phi = rng.dirichlet(alpha=np.ones(32) * sparsity, size=J)
    etaC = rng.dirichlet(alpha=np.ones(3) * sparsity, size=K)
    etaT = rng.dirichlet(alpha=np.ones(3) * sparsity, size=K)
    eta = np.stack([etaC, etaT], axis = 1)
    return phi, eta
