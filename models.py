from utils import *
from sklearn.cluster import k_means

def ch_dirichlet(node_name, a, shape, scale=1, testval = None):
    X = pm.Gamma(f'gamma_{node_name}', mu = a, sigma = scale, shape = shape, testval = testval)
    X = pm.Deterministic(node_name, (X/X.sum(axis = 1)[:,None]))
    return X

def reparam_factory(corpus, J: int, K: int, alpha_bias: float, psi_bias: float, 
                    gamma_bias: float, beta_bias: float, sig_obs = None, init_strategy = 'uniform'):
    
    S = corpus.shape[0]
    N = corpus.sum(1).reshape(S,1)
    phi_obs, etaC_obs, etaT_obs = unpack_sigs(sig_obs)
    phi_init, etaC_init, etaT_init = unpack_sigs(init_sigs(init_strategy, data=corpus, J=J, K=K))
    
    with pm.Model() as model:
        
        data = pm.Data("data", corpus)
        phi = ch_dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C), testval = phi_init)
        theta = ch_dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = ch_dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones((K,4)) * beta_bias
        etaC = pm.Dirichlet("etaC", a=beta[:,[0,2,3]], shape=(1, K, M), observed=etaC_obs, testval = etaC_init)
        etaT = pm.Dirichlet("etaT", a=beta[:,[0,1,2]], shape=(1, K, M), observed=etaT_obs, testval = etaT_init)
        eta = pm.Deterministic('eta', pm.math.concatenate([etaC, etaT], axis=0))
        
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
        lik = pm.Multinomial('corpus', n = N, p = B , observed=data)

    return model
    
def collapsed_model_factory(corpus, J: int, K: int, alpha_bias: float, psi_bias: float, 
                            gamma_bias: float, beta_bias: float, sig_obs = None, init_strategy = 'uniform'):
    
    S = corpus.shape[0]
    N = corpus.sum(1).reshape(S,1)
    phi_obs, etaC_obs, etaT_obs = unpack_sigs(sig_obs)
    phi_init, etaC_init, etaT_init = unpack_sigs(init_sigs(init_strategy, data=corpus, J=J, K=K))
    
    with pm.Model() as model:
        
        data = pm.Data("data", corpus)
        phi = pm.Dirichlet('phi', a = np.ones(C) * alpha_bias, shape=(J, C), observed=phi_obs, testval = phi_init)
        theta = pm.Dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = pm.Dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones((K,4)) * beta_bias
        etaC = pm.Dirichlet("etaC", a=beta[:,[0,2,3]], shape=(1, K, M), observed=etaC_obs, testval = etaC_init)
        etaT = pm.Dirichlet("etaT", a=beta[:,[0,1,2]], shape=(1, K, M), observed=etaT_obs, testval = etaT_init)
        eta = pm.Deterministic('eta', pm.math.concatenate([etaC, etaT], axis=0))

        B = pm.Deterministic("B", (pm.math.matrix_dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                   pm.math.matrix_dot(tt.batched_dot(theta,A),eta)[:,:,:,None]).reshape((S, -1)))
        
        # mutation counts
        lik = pm.Multinomial('corpus', n = N, p = B , observed=data)

    return model

def collapsed_model_factory_old_and_stinky(corpus, J: int, K: int, alpha_bias: float, psi_bias: float, 
                            gamma_bias: float, beta_bias: float, phi_obs = None, eta_obs = None, phi_init =None, eta_init = None):
    
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

def unpack_sigs(sigs_dict):
    # unpack a phi, eta to phi, etaC, etaT
    # assume eta is shape Kx2X3
    if sigs_dict is None: 
        return None, None, None
    else:
        if sigs_dict['eta'] is None:
            etaC = None
            etaT = None
        else: 
            etaC = sigs_dict['eta'][None,:,0,:]
            etaT = sigs_dict['eta'][None,:,1,:]
            
        return sigs_dict['phi'], etaC, etaT

    
    
def kmeans_alr(data, nsig, sigfunc):
    km = k_means(alr(sigfunc(data)), nsig)
    return alr_inv(km[0])
            
def init_kmeans(data=None, J=None, K=None):
    return {'phi': kmeans_alr(data, J, get_phis),
            'eta': kmeans_alr(data, K, get_etas).reshape(-1,2,3)} 
 
    
def init_from_tau(tau=None):
    # return phi and eta naively from tau
    # J = K = I
    return {'phi': get_phis(tau),
            'eta': get_etas(tau).reshape(-1,2,3)}


def init_sigs(strategy = 'kmeans', **kwargs):
    
    strats = ['kmeans', 'supply_tau', 'uniform']
    assert strategy in strats, f'strategy should be one of {strats}'
    
    if strategy == 'kmeans':
        return init_kmeans(**kwargs) 
    elif strategy == 'supply_tau':
        raise NotImplemented
        return init_from_tau(**kwargs)
    elif strategy == 'uniform':
        # default from pymc3
        return {'phi': None, 'eta': None} 
    
    

