# sim.py
from .utils import *

def sim_from_sigs(tau, tau_hyperprior, S, N, I=None, seed=None):
    """
    Simulate mutation data from predefined signatures.

    This function generates simulated mutation data based on given signatures
    and hyperparameters. It uses a Dirichlet process to generate sample-specific
    activities and then creates mutation counts for each sample.

    Parameters:
    -----------
    tau : pandas.DataFrame
        Predefined signatures in COSMIC format.
    tau_hyperprior : float
        Concentration parameter for the Dirichlet prior on signature activities.
    S : int
        Number of samples to simulate.
    N : int
        Number of mutations per sample.
    I : int, optional
        Number of signatures to use. If None, all signatures in tau are used.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    data : pandas.DataFrame
        Simulated mutation data. Each row represents a sample, and each column
        represents a mutation type.
    sim_params : dict
        Dictionary containing simulation parameters:
        - 'tau': The signatures used for simulation.
        - 'tau_activities': The generated sample-specific activities.

    """
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
    data = pd.DataFrame(data, columns = mut96, index = [f'simulated_sample_{n}' for n in range(S)])

    return data, {'tau':tau, 'tau_activities': tau_activities}

def sim_parametric(n_damage_sigs,n_misrepair_sigs,S,N,alpha_bias=0.9,psi_bias=0.1,gamma_bias=0.1,beta_bias=0.9,seed=1333):
    """
    Simulate data using a parametric model with damage and misrepair signatures.

    This function generates simulated mutation data based on a model with
    separate damage and misrepair processes. It creates random distributions
    for damage signatures (phi), sample-specific activities (theta),
    misrepair signatures (eta), and their interactions (A).

    Parameters:
    -----------
    n_damage_sigs : int
        Number of damage signatures to simulate.
    n_misrepair_sigs : int
        Number of misrepair signatures to simulate.
    S : int
        Number of samples to simulate.
    N : int
        Number of mutations per sample.
    alpha_bias : float, optional
        Concentration parameter for the Dirichlet prior on damage signatures (default: 0.9).
    psi_bias : float, optional
        Concentration parameter for the Dirichlet prior on sample-specific activities (default: 0.1).
    gamma_bias : float, optional
        Concentration parameter for the Dirichlet prior on misrepair signature activities (default: 0.1).
    beta_bias : float, optional
        Concentration parameter for the Dirichlet prior on misrepair signatures (default: 0.9).
    seed : int, optional
        Random seed for reproducibility (default: 1333).

    Returns:
    --------
    tuple
        A tuple containing two elements:
        1. pandas.DataFrame: Simulated mutation counts for each sample and mutation type.
        2. dict: Dictionary containing the generated model parameters and intermediate results.

    """
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
    """
    Encode mutation counts into a format suitable for topic modeling.

    This function takes a DataFrame of mutation counts and encodes it into two lists
    of indices: one for the 32 mutation types and another for the 6 possible base changes.

    Parameters
    ----------
    counts : pandas.DataFrame
        A DataFrame where each row represents a sample and each column represents
        a mutation type (96 mutation types in total).

    Returns
    -------
    tuple
        A tuple containing two elements:
        
        1. list of lists: Each inner list contains indices (0-31) representing the
           32 mutation types for each mutation in each sample.
        2. list of lists: Each inner list contains indices (0-5) representing the
           6 possible base changes for each mutation in each sample.

    Notes
    -----
    The encoding is based on the 96 mutation types, which are converted into
    32 mutation types (context) and 6 base changes. This encoding is useful
    for topic modeling approaches in mutation signature analysis.
    """
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


