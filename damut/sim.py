# sim_data.py
# simulate mutation catalogue using cosmic signatures
from utils import *

@extyaml
def sim_cosmic(sig_defs: str, S: int, N: int, tau_hyperprior: float, I: int, seed: int):
    # simulate from a predefined set of signatures (cosmic format)
    np.random.seed(seed)
    pm.set_tt_rng(seed)
    N = np.array([N] * S)
    tau = load_sigs(sig_defs, naming_style = 'cosmic', sep = '\t').to_numpy().T

    if I:
        tau = tau[np.random.choice(tau.shape[0], size = I, replace = False)]
    else: I = tau.shape[0]
        
    # draw activities according to tau
    tau_activities = pm.Dirichlet.dist(a=np.ones(I) * tau_hyperprior).random(size = S)
    B=(tau_activities @ tau)
    logging.debug(f'generated tau_activities shape {tau_activities.shape}')
    logging.debug(f'generated sigs (cosmic) shape {tau.shape}')
    corpus = np.vstack([d.random(size = 1) for d in map(pm.Multinomial.dist, N, B)])
    logging.debug(f'generated corpus shape {corpus.shape}')
    return corpus, tau, tau_activities

@extyaml
def sim_parametric(J:int,K:int,N:int,S:int,alpha_bias:float,psi_bias:float,
                   gamma_bias:float,beta_bias:float,seed:int):
    # simulate from generated phi and eta
    np.random.seed(seed)
    pm.set_tt_rng(seed)
    N = np.array([N] * S)

    # Hyper-parameter for priors
    alpha = np.ones(C) * alpha_bias
    psi = np.ones(J) * psi_bias
    gamma = np.ones(K) * gamma_bias
    beta = np.ones((K,4)) * beta_bias
    
    phi = pm.Dirichlet.dist(a=alpha, shape=(C)).random(size = J)
    theta = pm.Dirichlet.dist(a=psi).random(size = S)
    A = pm.Dirichlet.dist(a=gamma, shape=(J, K)).random(size = S)
    # ACGT
    # 0123
    eta = np.vstack([[pm.Dirichlet.dist(a=beta[:,[0,2,3]]).random(size=1)] , 
                         [pm.Dirichlet.dist(a=beta[:,[0,1,2]]).random(size=1)] ]).squeeze()

    W=np.dot(theta, phi).reshape(S,2,16)
    Q=np.einsum('sj,sjk,pkm->spm', theta, A, eta)
    B=np.einsum('spc,spm->spmc', W, Q).reshape(S, -1)
    
    data = np.vstack([d.random(size = 1) for d in map(pm.Multinomial.dist, N, B)])
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