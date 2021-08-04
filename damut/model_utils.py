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