from mimetypes import init
import pymc3 as pm
import numpy as np
from .utils import dirichlet, get_phi, get_eta
from .base import Damuta, DataSet
#from sklearn.decomposition import NMF
from theano.tensor import batched_dot
from sklearn.cluster import k_means

# note: model factories will likely be depricated in the future of pymc3
# see https://lucianopaz.github.io/2019/08/19/pymc3-shape-handling/


__all__ = ['Lda', 'TandemLda', 'HierarchicalTandemLda']

class Lda(Damuta):
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
                 alpha_bias=0.1, psi_bias=0.01,
                 opt_method="ADVI", seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, seed=seed)
        
        self.n_sigs = n_sigs
        self.model_kwargs = {"n_sigs": n_sigs, "alpha_bias": alpha_bias, "psi_bias": psi_bias}
        
    
    def _init_kmeans(self):
        # TODO: debug
        data=self.dataset.counts.to_numpy()
        
        # get proportions for signature initialization
        data = data/data.sum(1)[:,None]
        return k_means(data, self.n_sigs, init='k-means++', random_state=np.random.RandomState(self._rng.bit_generator))[0]
    
    
    def _initialize_signatures(self, init_strategy):
        super()._initialize_signatures(init_strategy)
        
        if init_strategy == "kmeans":
            self.model_kwargs['tau_init'] = self._init_kmeans()
        
        if init_strategy == "uniform":
            self.model_kwargs['tau_init'] = None   
    
    
    def _build_model(self, n_sigs, alpha_bias, psi_bias, tau_init):
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
            tau = dirichlet('tau', a = np.ones(96) * alpha_bias, shape=(I, 96), testval = tau_init)
            theta = dirichlet("theta", a = np.ones(I) * psi_bias, shape=(S, I))
            B = pm.Deterministic("B", pm.math.dot(theta, tau))
            # mutation counts
            pm.Multinomial('corpus', n = N, p = B, observed=data)
            
    def BOR(self):
        """Baysean Occam's Rasor"""
        pass
    
class TandemLda(Damuta):
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
                 opt_method="ADVI", seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, seed=seed)
         
        self.n_damage_sigs = n_damage_sigs
        self.n_misrepair_sigs = n_misrepair_sigs
        self.model_kwargs = {"n_damage_sigs": n_damage_sigs, "n_misrepair_sigs": n_misrepair_sigs, 
                             "alpha_bias": alpha_bias, "psi_bias": psi_bias,
                             "beta_bias": beta_bias, "gamma_bias": gamma_bias}
    
    
    def _init_kmeans(self):
        
        data=self.dataset.counts.to_numpy()
        
        # get proportions for signature initialization
        data = data/data.sum(1)[:,None]
        
        phi = k_means(get_phi(data), self.n_damage_sigs, init='k-means++',random_state=np.random.RandomState(self._rng.bit_generator))[0]
        eta = k_means(get_eta(data).reshape(-1,6), self.n_misrepair_sigs, init='k-means++', random_state=np.random.RandomState(self._rng.bit_generator))[0].reshape(-1,2,3)
      
        return phi, eta[:,0,:], eta[:,1,:]
    
    
    def _initialize_signatures(self, init_strategy):
        super()._initialize_signatures(init_strategy)
        
        if init_strategy == "kmeans":
            self.model_kwargs['phi_init'], \
                self.model_kwargs['etaC_init'], \
                    self.model_kwargs['etaT_init'] = self._init_kmeans()
        
        if init_strategy == "uniform":
            self.model_kwargs['phi_init'] = None
            self.model_kwargs['etaC_init'] = None 
            self.model_kwargs['etaT_init'] = None  
        
        # check that sigs are valid
        if self.model_kwargs["phi_init"] is not None:
            assert np.allclose(self.model_kwargs["phi_init"].sum(1), 1) 
        # eta should be kxpxm
        if self.model_kwargs["etaC_init"] is not None:
            assert np.allclose(self.model_kwargs["etaC_init"].sum(1), 1) 
        if self.model_kwargs["etaT_init"] is not None:
            assert np.allclose(self.model_kwargs["etaT_init"].sum(1), 1)       
    
    def _build_model(self, n_damage_sigs, n_misrepair_sigs, alpha_bias, psi_bias, beta_bias, gamma_bias,  
                     phi_init=None, etaC_init = None, etaT_init = None):
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
            
    def BOR(self):
        """Baysean Occam's Rasor"""
        pass


class HierarchicalTandemLda(Damuta):
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
                 opt_method="ADVI", seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, seed=seed)
        
        self.dataset.annotate_tissue_types(type_col)
        
        self.n_damage_sigs = n_damage_sigs
        self.n_misrepair_sigs = n_misrepair_sigs
        self.model_kwargs = {"n_damage_sigs": n_damage_sigs, "n_misrepair_sigs": n_misrepair_sigs, 
                             "alpha_bias": alpha_bias, "psi_bias": psi_bias,
                             "beta_bias": beta_bias}
    
    def _init_kmeans(self):
        
        data=self.dataset.counts.to_numpy()
        
        # get proportions for signature initialization
        data = data/data.sum(1)[:,None]
        
        phi = k_means(get_phi(data), self.n_damage_sigs, init='k-means++',random_state=np.random.RandomState(self._rng.bit_generator))[0]
        eta = k_means(get_eta(data).reshape(-1,6), self.n_misrepair_sigs, init='k-means++', random_state=np.random.RandomState(self._rng.bit_generator))[0].reshape(-1,2,3)
      
        return phi, eta[:,0,:], eta[:,1,:]
    
    
    def _initialize_signatures(self, init_strategy):
        super()._initialize_signatures(init_strategy)
        
        if init_strategy == "kmeans":
            self.model_kwargs['phi_init'], \
                self.model_kwargs['etaC_init'], \
                    self.model_kwargs['etaT_init'] = self._init_kmeans()
        
        if init_strategy == "uniform":
            self.model_kwargs['phi_init'] = None
            self.model_kwargs['etaC_init'] = None 
            self.model_kwargs['etaT_init'] = None  
        
        # check that sigs are valid
        if self.model_kwargs["phi_init"] is not None:
            assert np.allclose(self.model_kwargs["phi_init"].sum(1), 1) 
        # eta should be kxpxm
        if self.model_kwargs["etaC_init"] is not None:
            assert np.allclose(self.model_kwargs["etaC_init"].sum(1), 1) 
        if self.model_kwargs["etaT_init"] is not None:
            assert np.allclose(self.model_kwargs["etaT_init"].sum(1), 1)       
    
    def _build_model(self, n_damage_sigs, n_misrepair_sigs, alpha_bias, psi_bias, beta_bias, 
                     phi_init=None, etaC_init=None, etaT_init=None):
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
            phi = dirichlet('phi', a = np.ones(32) * alpha_bias, shape=(J, 32), testval = phi_init)
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
            
    def BOR(self):
        """Baysean Occam's Rasor"""
        pass
    

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



