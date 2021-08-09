from .utils import *
from sklearn.decomposition import NMF

# note: model factories will be depricated in the future of pymc3

def ch_dirichlet(node_name, a, shape, scale=1, testval = None):
    # dirichlet reparameterized here because of stickbreaking bug
    # https://github.com/pymc-devs/pymc3/issues/4733
    X = pm.Gamma(f'gamma_{node_name}', mu = a, sigma = scale, shape = shape, testval = testval)
    X = pm.Deterministic(node_name, (X/X.sum(axis = (X.ndim-1))[...,None]))
    return X

def tandem_lda(train, J, K, alpha_bias, psi_bias, gamma_bias, beta_bias, rng = np.random.default_rng(),
               phi_obs=None, etaC_obs=None, etaT_obs=None, init_strategy = 'uniform', tau = None, cbs=None):
    # latent dirichlet allocation with tandem signautres of damage and repair
    
    S = train.shape[0]
    N = train.sum(1).reshape(S,1)
    phi_init, etaC_init, etaT_init = da.init_sigs(init_strategy, data=train, J=J, K=K, tau=tau)
    
    with pm.Model() as model:
        
        data = pm.Data("data", train)
        phi = ch_dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C), testval = phi_init)
        theta = ch_dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = ch_dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones(4) * beta_bias
        etaC = ch_dirichlet("etaC", a=beta[[0,2,3]], shape=(K, M), testval = etaC_init)
        etaT = ch_dirichlet("etaT", a=beta[[0,1,2]], shape=(K, M), testval = etaT_init)
        eta = pm.Deterministic('eta', pm.math.stack([etaC, etaT], axis=1))

        B = pm.Deterministic("B", (pm.math.dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                   pm.math.dot(tt.batched_dot(theta,A), eta.dimshuffle(1,0,2))[:,:,:,None])).reshape((S, -1))
        
        # mutation counts
        pm.Multinomial('corpus', n = N, p = B, observed=data)

    return model


def tandtiss_lda():
    # latent dirichlet allocation with tandem signautres of damage and repair
    # and hirearchical tissue-specific priors
    raise NotImplemented
    
def vanilla_lda():
    raise NotImplemented

def vanilla_nmf(train, I):
    model = NMF(n_components=I, init='random', random_state=0)
    W = model.fit_transform(train)
    H = model.components_

def init_sigs(strategy, rng=np.random.default_rng(), data=None, J=None, K=None, tau=None):
    
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
    etaC, etaT = (eta[:,0,:], eta[:,1,:]) if eta is not None else (None, None)
    return phi, etaC, etaT
    
def init_kmeans(data, J, K, rng):
    if isinstance(data, pd.core.frame.DataFrame):
        data = data.to_numpy()
        
    return kmeans_alr(get_phi(data), J, rng), kmeans_alr(get_eta(data).reshape(-1,P*M), K, rng).reshape(-1,P,M)
 
def init_from_tau(tau, J, K, rng):
    # return phi and eta naively from tau
    # if I > J/K , I is randomly sampled
    # if I < J/K , taus will be  sampled with replacement
    
    if isinstance(tau, pd.core.frame.DataFrame):
        tau = tau.to_numpy()
    
    I = tau.shape[0]
    phi = rng.choice(get_phi(tau), size = J, replace = I>J)
    eta = rng.choice(get_eta(tau), size = K, replace = I>K)
    
    return phi, eta
   
def init_random(J, K, rng, sparsity = 0.1):
    # a random, but non-uniform initialization
    phi = rng.dirichlet(alpha=np.ones(C) * sparsity, size=J)
    etaC = rng.dirichlet(alpha=np.ones(M) * sparsity, size=K)
    etaT = rng.dirichlet(alpha=np.ones(M) * sparsity, size=K)
    eta = np.stack([etaC, etaT], axis = 1)
    return phi, eta
