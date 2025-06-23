import numpy as np
import pymc3 as pm
import pandas as pd
import warnings
from sklearn.cluster import k_means
from scipy.special import softmax, logsumexp, loggamma
from sklearn.metrics.pairwise import cosine_similarity
from .constants import * 
import pickle
import wandb
from scipy.optimize import linear_sum_assignment

# constants
#C=32
#M=3
#P=2

def dirichlet(node_name, a, shape, scale=1, testval=None, observed=None):
    """
    Create a reparameterized Dirichlet distribution using Gamma variables for use in PyMC3 models.

    Parameters
    ----------
    node_name : str
        Name for the node in the model.
    a : array-like
        Concentration parameters for the Dirichlet distribution.
    shape : tuple
        Shape of the resulting variable.
    scale : float, optional
        Scale parameter for the Gamma distribution (default: 1).
    testval : array-like, optional
        Test value for the variable.
    observed : array-like, optional
        Observed values for the variable.

    Returns
    -------
    pm.Deterministic
        A deterministic node representing the Dirichlet variable.
    """
    # dirichlet reparameterized here because of stickbreaking bug
    # https://github.com/pymc-devs/pymc3/issues/4733
    X = pm.Gamma(f'gamma_{node_name}', mu = a, sigma = scale, shape = shape, testval = testval, observed = observed)
    Y = pm.Deterministic(node_name, (X/X.sum(axis = (X.ndim-1))[...,None]))
    return Y

    
def save_checkpoint(fp, model, trace, dataset_args, model_args, pymc3_args, run_id): 
    with open(f'{fp}', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace, 'dataset_args': dataset_args, 
                     'model_args': model_args, 'pymc3_args': pymc3_args, 'run_id': run_id}, buff)
    print(f'checkpoint saved to {fp}') 
       
def load_checkpoint(fn):
    with open(fn, 'rb') as buff:
        data = pickle.load(buff)
        print(f'checkpoint loaded from {fn}') 
        wandb.init(id=data['run_id'], resume='allow')
        return data['model'], data['trace'], data['dataset_args'], data['model_args'], data['pymc3_args'], data['run_id'] 

def load_dataset(dataset_sel, counts_fp=None, annotation_fp=None, annotation_subset=None, seed=None,
                 data_seed = None, sig_defs_fp=None, sim_S=None, sim_N=None, sim_I=None, sim_tau_hyperprior=None,
                 sim_J=None, sim_K=None, sim_alpha_bias=None, sim_psi_bias=None, sim_gamma_bias=None, sim_beta_bias=None):
    # load counts, or simulated data - as specified by dataset_sel
    # seed -> rng as per https://albertcthomas.github.io/good-practices-random-number-generators/
    
    if dataset_sel == 'load_counts':
        dataset = load_counts(counts_fp)
        annotation = pd.read_csv(annotation_fp, index_col = 0, header = 0)
        dataset, annotation = subset_samples(dataset, annotation, annotation_subset)
        return dataset, annotation
        
    elif dataset_sel == 'sim_from_sigs':
        sig_defs = load_sigs(sig_defs_fp)
        dataset, sim_params = sim_from_sigs(sig_defs, sim_tau_hyperprior, sim_S, sim_N, sim_I, seed)
        return dataset, sim_params
    
    elif dataset_sel == 'sim_parametric':
        dataset, sim_params = sim_parametric(sim_J,sim_K,sim_S,sim_N,sim_alpha_bias,sim_psi_bias,sim_gamma_bias,sim_beta_bias,seed)
        return dataset, sim_params
    
    else:
        assert False, 'dataset selection not recognized'
    
def load_datasets(dataset_args):
    yargs = dataset_args.copy()
    ca = [load_dataset(counts_fp = j[0], annotation_fp = j[1], **yargs) for j in zip(yargs.pop('counts_fp'), yargs.pop('annotation_fp'))]
    counts = pd.concat([c[0] for c in ca ])
    annotation = pd.concat([a[1] for a in ca])
    return counts, annotation

def get_tau(phi, eta):
    """
    Compute the full 96-channel signature matrix from damage (phi) and misrepair (eta) signatures.

    Parameters
    ----------
    phi : np.ndarray
        Damage signatures, shape (n_damage, 32).
    eta : np.ndarray
        Misrepair signatures, shape (n_misrepair, 2, 3).

    Returns
    -------
    np.ndarray
        Combined signature matrix, shape (n_damage * n_misrepair, 96).
    """
    assert len(phi.shape) == 2 and len(eta.shape) == 3
    tau =  np.einsum('jpc,kpm->jkpmc', phi.reshape((-1,2,16)), eta).reshape((-1,96))
    return tau

def marginalize_for_phi(sigs):
    """
    Compute damage signatures (phi) by marginalizing over misrepair classes.

    Parameters
    ----------
    sigs : np.ndarray
        Signature matrix, shape (n_signatures, 96).

    Returns
    -------
    np.ndarray
        Damage signatures, shape (n_signatures, 32).
    """
    wrapped = sigs.reshape(-1, 2, 3, 16)
    phi = wrapped.sum(2).reshape(-1,32)
    return phi

def marginalize_for_eta(sigs, normalize=True):
    """
    Compute misrepair signatures (eta) by marginalizing over trinucleotide context classes.

    Parameters
    ----------
    sigs : np.ndarray
        Signature matrix, shape (n_signatures, 96).
    normalize : bool, optional
        Whether to normalize the output so each row sums to 1 (default: True).

    Returns
    -------
    np.ndarray
        Misrepair signatures, shape (n_signatures, 6).
    """
    wrapped = sigs.reshape(-1, 6, 16)
    eta = wrapped.sum(2).reshape(-1,3)
    if normalize:
        # normalize such that etaC and etaT sum to 1 respectively.
        eta = (eta/eta.sum(1)[:,None])
    return eta.reshape(-1,6)

def flatten_eta(eta):
    """
    Flatten a 3D eta array (p, k, m) to 2D (k, c) for compatibility.

    Parameters
    ----------
    eta : np.ndarray
        Misrepair signature array, shape (p, k, m).

    Returns
    -------
    np.ndarray
        Flattened array, shape (k, 6).
    """
    warnings.warn('Eta no longer constructed as pkm - use reshape instead', DeprecationWarning)
    return np.moveaxis(eta,0,1).reshape(-1, 6)

def alr(x, e=1e-12):
    """
    Compute the additive log-ratio (ALR) transformation for compositional data.

    Parameters
    ----------
    x : np.ndarray
        Input array, shape (n, d).
    e : float, optional
        Small value added for numerical stability (default: 1e-12).

    Returns
    -------
    np.ndarray
        ALR-transformed array, shape (n, d-1).
    """
    # add small value for stability in log
    x = x + e
    return (np.log(x) - np.log(x[...,-1]).reshape(-1,1))[:,0:-1]

def alr_inv(y):
    """
    Inverse additive log-ratio (ALR) transformation.

    Parameters
    ----------
    y : np.ndarray
        ALR-transformed array, shape (n, d-1).

    Returns
    -------
    np.ndarray
        Reconstructed compositional data, shape (n, d).
    """
    if y.ndim == 1: y = y.reshape(1,-1)
    return softmax(np.hstack([y, np.zeros((y.shape[0], 1)) ]), axis = 1)

def kmeans_alr(data, nsig, rng=np.random.default_rng()):
    """
    Perform k-means clustering in ALR space and return cluster centers in the original space.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_samples, n_features).
    nsig : int
        Number of clusters.
    rng : np.random.Generator, optional
        Random number generator (default: np.random.default_rng()).

    Returns
    -------
    np.ndarray
        Cluster centers in the original space, shape (nsig, n_features).
    """
    km = k_means(alr(data), nsig, random_state=np.random.RandomState(rng.bit_generator))
    return alr_inv(km[0])

def mult_ll(x, p):
    """
    Compute the multinomial log-likelihood for observed counts and probabilities.

    Parameters
    ----------
    x : np.ndarray
        Observed counts, shape (n_samples, n_categories).
    p : np.ndarray
        Probabilities, shape (n_samples, n_categories).

    Returns
    -------
    np.ndarray
        Log-likelihood values for each sample.
    """
    # Validate inputs first
    if x.shape != p.shape:
        raise ValueError(f"Shape mismatch: x.shape {x.shape} != p.shape {p.shape}")
    
    # Check for problematic zero probabilities  
    if np.any((p == 0) & (x > 0)):
        raise ValueError("Cannot compute log-likelihood: zero probability with non-zero count")
    
    # Original computation - let it warn if needed
    return loggamma(x.sum(1) + 1) - loggamma(x+1).sum(1) + (x * np.log(p)).sum(1)

def alp_B(data, B):
    """
    Compute the sum of multinomial log-likelihoods for a dataset and probability matrix.

    Parameters
    ----------
    data : np.ndarray
        Observed counts, shape (n_samples, n_categories).
    B : np.ndarray
        Probability matrix, shape (n_samples, n_categories).

    Returns
    -------
    float
        Total log-likelihood for the dataset.
    """
    return mult_ll(data, B).sum()

def lap_B(data, Bs):
    """
    Compute the log average posterior (LAP) for a dataset and a set of probability matrices.

    Parameters
    ----------
    data : np.ndarray
        Observed counts, shape (n_samples, n_categories).
    Bs : np.ndarray
        Array of probability matrices, shape (n_draws, n_samples, n_categories).

    Returns
    -------
    float
        Log average posterior value.
    """
    # Bs should be shape DxSx96 where D is the number of posterior samples
    # use logsumexp for stability
    assert Bs.ndim == 3, 'expected multiple trials for B'
    return logsumexp(np.vstack([mult_ll(data, B) for B in Bs]).sum(1)) - np.log(Bs.shape[0])

