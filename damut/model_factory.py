from .utils import *
from sklearn.cluster import k_means
from sklearn.decomposition import NMF

# note: model factories will be depricated in the future of pymc3

def ch_dirichlet(node_name, a, shape, scale=1, testval = None):
    # dirichlet reparameterized here because of stickbreaking bug
    # https://github.com/pymc-devs/pymc3/issues/4733
    X = pm.Gamma(f'gamma_{node_name}', mu = a, sigma = scale, shape = shape, testval = testval)
    X = pm.Deterministic(node_name, (X/X.sum(axis = 1)[:,None]))
    return X

def tandem_lda(train, J, K, alpha_bias, psi_bias, gamma_bias, beta_bias, 
               sig_obs = None, init_strategy = 'uniform', tau = None):
    # latent dirichlet allocation with tandem signautres of damage and repair
    
    S = train.shape[0]
    N = train.sum(1).reshape(S,1)
    phi_obs, etaC_obs, etaT_obs = unpack_sigs(sig_obs)
    phi_init, etaC_init, etaT_init = init_sigs(init_strategy, data=train, J=J, K=K, tau=tau)
    
    with pm.Model() as model:
        
        data = pm.Data("data", train)
        phi = ch_dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C), testval = phi_init)
        theta = ch_dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = ch_dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones(4) * beta_bias
        etaC = ch_dirichlet("etaC", a=beta[[0,2,3]], shape=(K, M), testval = etaC_init)
        etaT = ch_dirichlet("etaT", a=beta[[0,1,2]], shape=(K, M), testval = etaT_init)
        tt.printing.Print()(etaC)
        tt.printing.Print()(etaT)
        eta = pm.Deterministic('eta', pm.math.stack([etaC, etaT], axis=1))
        tt.printing.Print()(eta[1])

        B = pm.Deterministic("B", (pm.math.matrix_dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                   pm.math.matrix_dot(tt.batched_dot(theta,A),eta)[:,:,:,None]).reshape((S, -1)))
        
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

def init_sigs(strategy, rng, data=None, J=None, K=None, tau=None):
    
    strats = ['kmeans', 'supply_tau', 'uniform']
    assert strategy in strats, f'strategy should be one of {strats}'
    
    if strategy == 'kmeans':
        phi, eta = init_kmeans(data, J, K, rng) 
    elif strategy == 'supply_tau':
        phi, eta = init_from_tau(tau, J, K, rng)
    elif strategy == 'uniform':
        # default from pymc3
        phi, eta = None, None
    
    # eta should be kxpxm
    etaC, etaT = eta[:,0,:], eta[:,1,:] if eta else None, None
    return phi, lambda e: None, None if e else None, None
    
def init_kmeans(data, J, K):
    return {'phi': kmeans_alr(data, J, get_phi, rng),
            'eta': kmeans_alr(data, K, get_eta, rng)} 
 
def init_from_tau(tau, J, K):
    # return phi and eta naively from tau
    # if I > J/K , I is randomly sampled
    # if I < J/K , taus will be  sampled with replacement
    
    if isinstance(tau, pd.core.frame.DataFrame):
        tau = tau.to_numpy()
    
    I = tau.shape[0]
    phi = rng.choice(get_phi(tau), size = J, replace = I>J)
    eta = rng.choice(get_eta(tau), size = K, replace = I>K)
    
    return {'phi': phi, 'eta': eta}
   
def init_random():
    raise NotImplemented
