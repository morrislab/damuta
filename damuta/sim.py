# sim.py
from .utils import *

def sim_from_sigs(tau, tau_hyperprior, S, N, I=None, seed=None):
    # simulate from a predefined set of signatures (cosmic format)
    rng=np.random.default_rng(seed)
    
    if I:
        tau = tau.sample(n=I, random_state = rng.bit_generator)
    else: I = tau.shape[0]
        
    # draw activities according to tau
    tau_activities = rng.dirichlet(alpha=np.ones(I) * tau_hyperprior, size=S)
    B=(tau_activities @ tau)
    
    # fix B if cast to df from tau
    if isinstance(B, pd.core.frame.DataFrame):
        B = B.to_numpy()

    data = np.vstack(list(map(rng.multinomial, [N]*S, B, [1]*S)))
    data = pd.DataFrame(data, columns = mut96, index = [f'simsample_{n}' for n in range(S)])

    return data, {'tau':tau, 'tau_activities': tau_activities}

def sim_parametric(n_damage_sigs,n_misrepair_sigs,S,N,alpha_bias=0.01,psi_bias=0.01,gamma_bias=0.01,beta_bias=0.01,seed=1333):
    # simulate from generated phi and eta
    J=n_damage_sigs
    K=n_misrepair_sigs
    rng=np.random.default_rng(seed)
    
    # Hyper-parameter for priors
    alpha = np.ones(32) * alpha_bias
    psi = np.ones(J) * psi_bias
    gamma = np.ones(K) * gamma_bias
    beta = np.ones((K,4)) * beta_bias
    # ACGT
    # 0123
    beta = np.vstack([beta[:,[0,2,3]], beta[:,[0,1,2]]])
    
    phi = rng.dirichlet(alpha=alpha, size=J) 
    theta = rng.dirichlet(alpha=psi, size=S) 
    A = rng.dirichlet(alpha=gamma, size=(S,J)) 

    eta = np.vstack(list(map(rng.dirichlet, beta))).reshape(2,K,3)
    
    W=np.dot(theta, phi).reshape(S,2,16)
    Q=np.einsum('sj,sjk,pkm->spm', theta, A, eta)
    B=np.einsum('spc,spm->spmc', W, Q).reshape(S, -1)
    
    data = np.vstack(list(map(rng.multinomial, [N]*S, B, [1]*S)))
    data = pd.DataFrame(data, columns = mut96, index = [f'simulated_sample_{n}' for n in range(S)])
    
    return data, {'phi': phi, 'theta': theta, 'A': A, 'eta': eta, 'B': B,
                  'alpha': alpha, 'psi': psi, 'gamma': gamma, 'beta': beta}

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