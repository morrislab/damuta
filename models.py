from utils import *

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