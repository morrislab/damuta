import pymc3 as pm
import numpy as np
import random
from scipy.special import logsumexp
from .utils import dirichlet, get_phi, get_eta
from .base import Model, DataSet
from .plotting import *
#from sklearn.decomposition import NMF
from theano.tensor import batched_dot
from sklearn.cluster import k_means

# note: model factories will likely be depricated in the future of pymc3
# see https://lucianopaz.github.io/2019/08/19/pymc3-shape-handling/


__all__ = ['Lda', 'TandemLda', 'HierarchicalTandemLda']

class Lda(Model):
    """Bayesian inference of mutational signautres and their activities.
    
    Fit COSMIC-style mutational signatures with a Latent Dirichlet Allocation model. 
    
    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    n_sigs: int
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
    
    def __init__(self, dataset: DataSet, n_sigs: int,
                 alpha_bias=0.1, psi_bias=0.01, tau_obs=None,
                 opt_method="ADVI", init_strategy="uniform",
                 init_signatures=None, seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, init_strategy=init_strategy, init_signatures=init_signatures, seed=seed)
        self.n_sigs = n_sigs
        self._model_kwargs = {"n_sigs": n_sigs, "alpha_bias": alpha_bias, "psi_bias": psi_bias, 'tau_obs': tau_obs}
        
    def _init_uniform(self):
        self._model_kwargs['tau_init'] = None 
        
    def _init_kmeans(self):
        data=self.dataset.counts.to_numpy().copy()
        # add pseudo count to 0 categories
        data[data==0] = 1
        # get proportions for signature initialization
        data = data/data.sum(1)[:,None]
        self._model_kwargs['tau_init'] = k_means(data, self.n_sigs, init='k-means++', random_state=np.random.RandomState(self._rng.bit_generator))[0]
    
    def _init_from_sigs(self):
        if self.n_sigs != self.init_signatures.n_sigs:
            warnings.warn(f'init_signatures signature dimension does not match n_sigs of {self.n_sigs}. Argument n_sigs will be ignored.')
            self.n_sigs = self.init_signatures.n_sigs
            self._model_kwargs['n_sigs'] = self.init_signatures.n_sigs
        tau = self.init_signatures.signatures.to_numpy()
        # add pseudo count to support and renormalize 
        tau[np.isclose(tau, 0)] = tau[np.isclose(tau, 0)] + 1e-7
        tau = tau/tau.sum(1)[:,None]
        self._model_kwargs['tau_init'] = tau

    def _build_model(self, n_sigs, alpha_bias, psi_bias, tau_init=None, tau_obs=None):
        """Compile a pymc3 model
        
        Parameters 
        ----------
        n_sigs: int
            Number of signautres to fit
        alpha_bias: float, or numpy array of shape (96,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of mutation types appearing in inferred signatures
        psi_bias: float, or numpy array of shape (n_sigs,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of each signature activities
        tau_init: numpy array of shape (n_sigs, 96)
            Signatures to initialize inference with 
        """
        data=self.dataset.counts.to_numpy()
        S = data.shape[0]
        N = data.sum(1).reshape(S,1)
        I = n_sigs
        
        with pm.Model() as self.model:
            
            data = pm.Data("data", data)
            tau = dirichlet('tau', a = np.ones(96) * alpha_bias, shape=(I, 96), testval = tau_init, observed = tau_obs)
            theta = dirichlet("theta", a = np.ones(I) * psi_bias, shape=(S, I))
            B = pm.Deterministic("B", pm.math.dot(theta, tau))
            # mutation counts
            pm.Multinomial('corpus', n = N, p = B, observed=data)
    
class TandemLda(Model):
    """Bayesian inference of mutational signautres and their activities.
    
    Fit COSMIC-style mutational signatures with a Tandem LDA model, where damage signatures
    and misrepair signatures each have their own set of activities. 
    
    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    n_damage_sigs: int
        Number of damage signautres to fit
    n_misrepair_sigs: int
        Number of misrepair signatures to fit
    alpha_bias: float, or numpy array of shape (32,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of trinucleotide context types appearing in inferred damage signatures
    psi_bias: float, or numpy array of shape (n_damage_sigs,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of each damage signature activities
    beta_bias: float, or numpy array of shape (6,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of substitution types appearing in inferred damage signatures
    gamma_bias: float, or numpy array of shape (n_missrepair_sigs,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of each misrepair signature activities
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
    
    def __init__(self, dataset: DataSet, n_damage_sigs: int, n_misrepair_sigs: int,
                 alpha_bias=0.1, psi_bias=0.01, beta_bias=0.1, gamma_bias=0.01,
                 phi_obs = None, etaC_obs = None, etaT_obs = None, 
                 opt_method="ADVI", init_strategy="kmeans", init_signatures=None, seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, init_strategy=init_strategy, init_signatures=init_signatures, seed=seed)
        self.n_damage_sigs = n_damage_sigs
        self.n_misrepair_sigs = n_misrepair_sigs 
        self._model_kwargs = {"n_damage_sigs": n_damage_sigs, "n_misrepair_sigs": n_misrepair_sigs, 
                             "alpha_bias": alpha_bias, "psi_bias": psi_bias,
                             "beta_bias": beta_bias, "gamma_bias": gamma_bias, 
                             "phi_obs": phi_obs, "etaC_obs": etaC_obs, "etaT_obs": etaT_obs}
    
    def _init_uniform(self):
        self._model_kwargs['phi_init'] = None
        self._model_kwargs['etaC_init'] = None 
        self._model_kwargs['etaT_init'] = None 
        
    def _init_kmeans(self):
        data=self.dataset.counts.to_numpy().copy()
        # add pseudo count to 0 categories
        data[data==0] = 1
        # get proportions for signature initialization
        data = data/data.sum(1)[:,None]
        self._model_kwargs['phi_init'] = k_means(get_phi(data), self.n_damage_sigs, init='k-means++',random_state=np.random.RandomState(self._rng.bit_generator))[0]
        eta = k_means(get_eta(data).reshape(-1,6), self.n_misrepair_sigs, init='k-means++', random_state=np.random.RandomState(self._rng.bit_generator))[0].reshape(-1,2,3)
        self._model_kwargs['etaC_init'] = eta[:,0,:]
        self._model_kwargs['etaT_init'] = eta[:,1,:]
    
    def _init_from_sigs(self):
        if self.n_damage_sigs != self.init_signatures.n_damage_sigs:
            warnings.warn(f'init_signatures damage dimension does not match n_damage_sigs of {self.n_damage_sigs}. Argument n_damage_sigs will be ignored.')
            self.n_damage_sigs = self.init_signatures.n_damage_sigs
            self._model_kwargs['n_damage_sigs'] = self.init_signatures.n_damage_sigs
        if self.n_misrepair_sigs != self.init_signatures.n_misrepair_sigs:
            warnings.warn(f'init_signatures misrepair dimension does not match n_misrepair_sigs of {self.n_misrepair_sigs}. Argument n_misrepair_sigs will be ignored.')
            self.n_misrepair_sigs = self.init_signatures.n_misrepair_sigs
            self._model_kwargs['n_misrepair_sigs'] = self.init_signatures.n_misrepair_sigs
        phi = self.init_signatures.damage_signatures.to_numpy()
        eta = self.init_signatures.misrepair_signatures.to_numpy().reshape(-1,2,3)
        # add pseudo count to support and renormalize 
        phi[np.isclose(phi, 0)] = phi[np.isclose(phi, 0)] + 1e-7
        phi = phi/phi.sum(1)[:,None]
        eta[np.isclose(eta, 0)] = eta[np.isclose(eta, 0)] + 1e-7
        eta = eta/eta.sum(2)[:,:,None]

        self._model_kwargs['phi_init'] = phi
        self._model_kwargs['etaC_init'] = eta[:,0,:]
        self._model_kwargs['etaT_init'] = eta[:,1,:]
           
    
    def _build_model(self, n_damage_sigs, n_misrepair_sigs, alpha_bias, psi_bias, beta_bias, gamma_bias,  
                     phi_init=None, etaC_init = None, etaT_init = None, phi_obs=None, etaC_obs=None, etaT_obs=None):
        """Compile a pymc3 model
        
        Parameters 
        ----------
        n_damage_sigs: int
            Number of damage signautres to fit
        n_misrepair_sigs: int
            Number of misrepair signautres to fit
        alpha_bias: float, or numpy array of shape (32,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of trinucleotide context types appearing in inferred damage signatures
        psi_bias: float, or numpy array of shape (n_damage_sigs,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of each damage signature activities
        beta_bias: float, or numpy array of shape (6,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of substitution types appearing in inferred damage signatures
        gamma_bias: float, or numpy array of shape (n_missrepair_sigs,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of each misrepair signature activities
        phi_init: numpy array of shape (n_damage_sigs, 32)
            Damage signatures to initialize inference with 
        etaC_init: numpy array of shape (n_misrepair_sigs, 3)
            C-context misrepair signatures to initialize inference with 
        etaT_init: numpy array of shape (n_misrepair_sigs, 3)
            T-context misrepair signatures to initialize inference with 
        """
        # latent dirichlet allocation with tandem signautres of damage and repair
        train = self.dataset.counts.to_numpy()
        
        S = train.shape[0]
        N = train.sum(1).reshape(S,1)
        J = n_damage_sigs
        K = n_misrepair_sigs
        
        with pm.Model() as self.model:
            
            data = pm.Data("data", train)
            phi = dirichlet('phi', a = np.ones(32) * alpha_bias, shape=(J, 32), testval = phi_init)
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

    
class HierarchicalTandemLda(TandemLda):
    """Bayesian inference of mutational signautres and their activities.
    
    Fit COSMIC-style mutational signatures with a Hirearchical Tandem LDA model, where damage signatures
    and misrepair signatures each have their own set of activities. A tissue-type hirearchical 
    prior is fit over damage-misrepair signature associations, for improved interpretability of 
    misrepair activity specificities. 
    
    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    n_damage_sigs: int
        Number of damage signautres to fit
    n_misrepair_sigs: int
        Number of misrepair signatures to fit
    type_col: str
        The name of the annotation column that holds the tissue type of each sample
    alpha_bias: float, or numpy array of shape (32,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of trinucleotide context types appearing in inferred damage signatures
    psi_bias: float, or numpy array of shape (n_damage_sigs,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of each damage signature activities
    beta_bias: float, or numpy array of shape (6,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of substitution types appearing in inferred damage signatures
    gamma_bias: float, or numpy array of shape (n_missrepair_sigs,)
        Dirichlet concentration parameter on (0,inf). Determines prior probability of each misrepair signature activities
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
    
    def __init__(self, dataset: DataSet, n_damage_sigs: int, n_misrepair_sigs: int,
                 type_col: str, alpha_bias=0.1, psi_bias=0.01, beta_bias=0.1,  
                 phi_obs = None, etaC_obs = None, etaT_obs = None,
                 opt_method="ADVI", init_strategy="kmeans", init_signatures=None, seed=2021):
        
        
        super().__init__(dataset=dataset, n_damage_sigs = n_damage_sigs, n_misrepair_sigs = n_misrepair_sigs, 
                         # TODO : fix hyperprior bug by uncommenting following line
                         #alpha_bias = alpha_bias, phi_bias = psi_bias, beta_bias = beta_bias,
                         phi_obs = phi_obs, etaC_obs = etaC_obs, etaT_obs = etaT_obs,
                         opt_method=opt_method, init_strategy=init_strategy, init_signatures=init_signatures, seed=seed)
        self._model_kwargs.pop('gamma_bias')
        self.dataset.annotate_tissue_types(type_col) 

    def _build_model(self, n_damage_sigs, n_misrepair_sigs, alpha_bias, psi_bias, beta_bias, 
                     phi_init=None, etaC_init=None, etaT_init=None,
                     phi_obs=None, etaC_obs=None, etaT_obs=None):
        """Compile a pymc3 model
        
        Parameters 
        ----------
        n_damage_sigs: int
            Number of damage signautres to fit
        n_misrepair_sigs: int
            Number of misrepair signautres to fit
        alpha_bias: float, or numpy array of shape (32,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of trinucleotide context types appearing in inferred damage signatures
        psi_bias: float, or numpy array of shape (n_damage_sigs,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of each damage signature activities
        beta_bias: float, or numpy array of shape (6,)
            Dirichlet concentration parameter on (0,inf). Determines prior probability of substitution types appearing in inferred damage signatures
        phi_init: numpy array of shape (n_damage_sigs, 32)
            Damage signatures to initialize inference with 
        etaC_init: numpy array of shape (n_misrepair_sigs, 3)
            C-context misrepair signatures to initialize inference with 
        etaT_init: numpy array of shape (n_misrepair_sigs, 3)
            T-context misrepair signatures to initialize inference with 
        """
        # latent dirichlet allocation with tandem signautres of damage and repair
        # and hirearchical tissue-specific priors
        
        train = self.dataset.counts.to_numpy()
        type_codes = self.dataset.type_codes
        
        S = train.shape[0]
        N = train.sum(1).reshape(S,1)
        J = n_damage_sigs
        K = n_misrepair_sigs
        
        with pm.Model() as self.model:
            
            data = pm.Data("data", train)
            phi = dirichlet('phi', a = np.ones(32) * alpha_bias, shape=(J, 32), testval = phi_init, observed=phi_obs)
            theta = dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
            
            a_t = pm.Gamma('a_t',1,1,shape = (max(type_codes + 1),K))
            b_t = pm.Gamma('b_t',1,1,shape = (max(type_codes + 1),K))
            g = pm.Gamma('gamma', alpha = a_t[type_codes], beta = b_t[type_codes], shape = (S,K))
            A = dirichlet("A", a = g, shape = (J, S, K)).dimshuffle(1,0,2)

            # 4 is constant for ACGT
            beta = np.ones(4) * beta_bias
            etaC = dirichlet("etaC", a=beta[[0,2,3]], shape=(K,3), testval = etaC_init, observed = etaC_obs)
            etaT = dirichlet("etaT", a=beta[[0,1,2]], shape=(K,3), testval = etaT_init, observed = etaT_obs)
            eta = pm.Deterministic('eta', pm.math.stack([etaC, etaT], axis=1)).dimshuffle(1,0,2)

            B = pm.Deterministic("B", (pm.math.dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                    pm.math.dot(batched_dot(theta,A), eta)[:,:,:,None]).reshape((S, -1)))
            
            # mutation counts
            pm.Multinomial('corpus', n = N, p = B, observed=data)

        

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



