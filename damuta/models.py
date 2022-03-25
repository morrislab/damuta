import pymc3 as pm
import numpy as np
from .utils import dirichlet
from .initialization import init_sigs
from .base import Damuta, DataSet
#from sklearn.decomposition import NMF
from theano.tensor import batched_dot

# note: model factories will likely be depricated in the future of pymc3
# see https://lucianopaz.github.io/2019/08/19/pymc3-shape-handling/


__all__ = ['Lda', 'TandemLda', 'HierarchicalTendemLda']

class Lda(Damuta):
    """Bayesian inference of mutational signautres and their activities.
    
    Fit COSMIC-style mutational signatures with a Latent Dirichlet Allocation model. 
    
    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    I: int
        Number of signautres to fit
    alpha_bias: float, or numpy array of shape (96,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of mutation types appearing in inferred signatures
    psi_bias: float, or numpy array of shape (I,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of each signature activities
    opt_method: str 
        one of "ADVI" for mean field inference, or "FullRankADVI" for full rank inference.
    seed : int
        Random seed 
    
    Attributes
    ----------
    model:
        pymc3 model instance
    model_kwargs: dict
        dict of parameters to pass when constructing model (ex. hyperprior values)
    approx:
        pymc3 approximation object. Created via self.fit()
    run_id: str
        Unique label used to identify run. Used when saving checkpoint files, drawn from wandb run if wandb is enabled.
    """
    
    def __init__(self, dataset: DataSet,
                 I: int, alpha_bias=0.1, psi_bias=0.01,
                 opt_method="ADVI", seed=2021):
        
        super(Damuta, self).__init__(dataset, opt_method, seed)

        np.random_seed(self.seed)
        pm.set_tt_rng(self.seed)
        
        self.model_kwargs = {"I": I, "alpha_bias": alpha_bias, "psi_bias": psi_bias}
    
    def build_model(self, I, alpha_bias, psi_bias):
        """Compile a pymc3 model
        
        Parameters 
        ----------
        I: int
            Number of signautres to fit
        alpha_bias: float, or numpy array of shape (96,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of mutation types appearing in inferred signatures
        psi_bias: float, or numpy array of shape (I,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of each signature activities
        """
        
        S = self.dataset.shape[0]
        N = self.dataset.sum(1).reshape(S,1)
        
        with pm.Model() as self.model:
            
            data = pm.Data("data", self.dataset)
            tau = dirichlet('tau', a = np.ones(96) * alpha_bias, shape=(I, 96))
            theta = dirichlet("theta", a = np.ones(I) * psi_bias, shape=(S, I))
            B = pm.Deterministic("B", pm.math.dot(theta, tau))
            # mutation counts
            pm.Multinomial('corpus', n = N, p = B, observed=data)
    
class TandemLda(Damuta):
    """foo bar"""
    
    a=3


class HierarchicalTendemLda(Damuta):
    """foo bar"""
    
    a=3

def tandem_lda(train, J, K, alpha_bias, psi_bias, gamma_bias, beta_bias, model_seed = 42, 
               phi_obs=None, etaC_obs=None, etaT_obs=None, init_strategy = 'uniform', tau = None, cbs=None):
    # latent dirichlet allocation with tandem signautres of damage and repair

    S = train.shape[0]
    N = train.sum(1).reshape(S,1)
    phi_init, etaC_init, etaT_init = init_sigs(init_strategy, data=train, J=J, K=K, tau=tau, seed=model_seed)
    
    with pm.Model() as model:
        
        data = pm.Data("data", train)
        phi = dirichlet('phi', a = np.ones(32) * alpha_bias, shape=(J, 3), testval = phi_init)
        theta = dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        A = dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
        # 4 is constant for ACGT
        beta = np.ones(4) * beta_bias
        etaC = dirichlet("etaC", a=beta[[0,2,3]], shape=(K, 3), testval = etaC_init)
        etaT = dirichlet("etaT", a=beta[[0,1,2]], shape=(K, 3), testval = etaT_init)
        eta = pm.Deterministic('eta', pm.math.stack([etaC, etaT], axis=1))

        B = pm.Deterministic("B", (pm.math.dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                   pm.math.dot(batched_dot(theta,A), eta.dimshuffle(1,0,2))[:,:,:,None]).reshape((S, -1)))
        
        # mutation counts
        pm.Multinomial('corpus', n = N, p = B, observed=data)

    return model

def tandtiss_lda(train, J, K, alpha_bias, psi_bias, gamma_bias, beta_bias, 
                 type_codes, model_seed=42, init_strategy = 'uniform', tau = None, cbs=None):
    # latent dirichlet allocation with tandem signautres of damage and repair
    # and hirearchical tissue-specific priors
    
    S = train.shape[0]
    N = train.sum(1).reshape(S,1)
    phi_init, etaC_init, etaT_init = init_sigs(init_strategy, data=train, J=J, K=K, tau=tau, seed=model_seed)
    
    with pm.Model() as model:
        
        data = pm.Data("data", train)
        phi = dirichlet('phi', a = np.ones(32) * alpha_bias, shape=(J, 3), testval = phi_init)
        theta = dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
        
        a_t = pm.Gamma('a_t',1,1,shape = (max(type_codes + 1),K))
        b_t = pm.Gamma('b_t',1,1,shape = (max(type_codes + 1),K))
        g = pm.Gamma('gamma', alpha = a_t[type_codes], beta = b_t[type_codes], shape = (S,K))
        A = dirichlet("A", a = g, shape = (J, S, K)).dimshuffle(1,0,2)

        # 4 is constant for ACGT
        beta = np.ones(4) * beta_bias
        etaC = dirichlet("etaC", a=beta[[0,2,3]], shape=(K,3), testval = etaC_init)
        etaT = dirichlet("etaT", a=beta[[0,1,2]], shape=(K,3), testval = etaT_init)
        eta = pm.Deterministic('eta', pm.math.stack([etaC, etaT], axis=1)).dimshuffle(1,0,2)

        B = pm.Deterministic("B", (pm.math.dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                   pm.math.dot(batched_dot(theta,A), eta)[:,:,:,None]).reshape((S, -1)))
        
        # mutation counts
        pm.Multinomial('corpus', n = N, p = B, observed=data)

    return model
    


def vanilla_nmf(train, I):
    """
    Examples
    --------
    >>> import pandas as pd
    >>> from damuta.constants import mut96
    >>> counts = pd.read_csv('data/pcawg_counts.csv', index_col=0, header=0)[mut96].to_numpy()
    >>> nmf = NMF(n_components=20, init='random', random_state=0)
    >>> w = nmf.fit_transform(counts)
    >>> h = nmf.components_
    >>> w_norm = nmf.fit_transform(counts/counts.sum(axis=1, keepdims=True))
    >>> h_norm = nmf.components_
    """
    
    
    raise NotImplemented
    #model = NMF(n_components=I, init='random', random_state=0)
    #W = model.fit_transform(train)
    #H = model.components_



