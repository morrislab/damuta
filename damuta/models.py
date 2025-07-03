import pymc3 as pm
import numpy as np
import pandas as pd
import warnings
from .utils import dirichlet, marginalize_for_phi, marginalize_for_eta
from .base import Model, DataSet, SignatureSet
from .constants import mut96, mut32, mut6
from theano.tensor import batched_dot
from sklearn.cluster import k_means

# note: model factories will likely be depricated in the future of pymc3
# see https://lucianopaz.github.io/2019/08/19/pymc3-shape-handling/


__all__ = ['Lda', 'TandemLda', 'HierarchicalTandemLda']

class Lda(Model):
    """Bayesian inference of mutational signatures and their activities.
    
    Fit COSMIC-style mutational signatures using a Latent Dirichlet Allocation (LDA) model.
    
    Parameters
    ----------
    dataset : DataSet
        Data object containing mutation counts for fitting.
    n_sigs : int
        Number of signatures to infer.
    alpha_bias : float or numpy.ndarray of shape (96,), default=0.1
        Dirichlet concentration parameter for signature prior. Controls the sparsity of inferred signatures.
    psi_bias : float or numpy.ndarray of shape (n_sigs,), default=0.01
        Dirichlet concentration parameter for signature activity prior. Controls the sparsity of signature activities.
    tau_obs : numpy.ndarray, optional
        Observed signatures to include in the model.
    opt_method : {'ADVI', 'FullRankADVI'}, default='ADVI'
        Optimization method for variational inference. 'ADVI' for mean-field, 'FullRankADVI' for full-rank.
    init_strategy : {'uniform', 'kmeans', 'from_sigs'}, default='uniform'
        Strategy for initializing signatures.
    init_signatures : SignatureSet, optional
        Pre-defined signatures for initialization when init_strategy is 'from_sigs'.
    seed : int, default=2021
        Random seed for reproducibility.
    
    Attributes
    ----------
    model : pymc3.Model
        PyMC3 model instance.
    model_kwargs : dict
        Dictionary of parameters used in model construction.
    approx : pymc3.approximations.Approximation
        Variational approximation object. Created after calling self.fit().
    run_id : str
        Unique identifier for the current run. Used for checkpoint files and wandb logging.
    n_sigs : int
        Number of signatures being fit.
    dataset : DataSet
        Input dataset used for fitting.
    
    Methods
    -------
    fit(n_iter=30000, **kwargs)
        Fit the model to the data using variational inference.
    sample_posterior(n_samples=1000)
        Sample from the fitted posterior distribution.
    get_signatures()
        Extract inferred signatures from the fitted model.
    get_activities()
        Extract inferred signature activities from the fitted model.
    
    """
    
    def __init__(self, dataset: DataSet, n_sigs=None,
                 alpha_bias=0.1, psi_bias=0.01, tau_obs=None,
                 opt_method="ADVI", init_strategy="kmeans",
                 init_signatures=None, seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, init_strategy=init_strategy, init_signatures=init_signatures, seed=seed)
        
        if n_sigs is None and tau_obs is None:
            raise ValueError("n_sigs must be provided when tau_obs is None")
        
        if tau_obs is not None:
            if n_sigs is not None:
                warnings.warn("argument n_sigs ({}) is ignored when tau_obs is provided".format(n_sigs))
            n_sigs = tau_obs.shape[0]

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
    

    def get_estimated_signatures(self, n_draws=1):
        """Extract signatures from model posterior"""
        self.check_is_fitted()
        hat = self.approx.sample(n_draws)
        return hat.tau

    def get_estimated_SignatureSet(self, n_draws=1):
        """Construct SignatureSet from posterior"""
        self.check_is_fitted()
        hat = self.approx.sample(n_draws)
        if n_draws > 1:
            warnings.warn(f"Signatures will be summarized as the mean of {n_draws} posterior samples")
        signatures = pd.DataFrame(hat.tau.mean(0), columns=mut96)
        signatures.index = [f'S{i+1}' for i in range(self.n_sigs)]
        return SignatureSet(signatures)

    def get_estimated_activities_DataFrame(self, n_draws=1):
        """Extract activities as DataFrame"""
        self.check_is_fitted()
        hat = self.approx.sample(n_draws)
        if n_draws > 1:
            warnings.warn(f"Activities will be summarized as the mean of {n_draws} posterior samples")
        return pd.DataFrame(hat.theta.mean(0), 
                        index=self.dataset.ids, 
                        columns=[f'S{i+1}' for i in range(self.n_sigs)])
    
class TandemLda(Model):
    """Bayesian inference of mutational signatures and their activities using a Tandem LDA model.
    
    This class fits COSMIC-style mutational signatures using a Tandem Latent Dirichlet Allocation (LDA) model,
    where damage signatures and misrepair signatures each have their own set of activities.
    
    Parameters
    ----------
    dataset : DataSet
        Data for fitting the model.
    n_damage_sigs : int
        Number of damage signatures to fit.
    n_misrepair_sigs : int
        Number of misrepair signatures to fit.
    alpha_bias : float or numpy.ndarray of shape (32,)
        Dirichlet concentration parameter on (0, inf) for damage signatures. Determines the prior probability
        of trinucleotide context types appearing in inferred damage signatures.
    psi_bias : float or numpy.ndarray of shape (n_damage_sigs,)
        Dirichlet concentration parameter on (0, inf) for damage signature activities. Determines the prior
        probability of each damage signature activity.
    beta_bias : float or numpy.ndarray of shape (6,)
        Dirichlet concentration parameter on (0, inf) for misrepair signatures. Determines the prior probability
        of substitution types appearing in inferred misrepair signatures.
    gamma_bias : float or numpy.ndarray of shape (n_misrepair_sigs,)
        Dirichlet concentration parameter on (0, inf) for misrepair signature activities. Determines the prior
        probability of each misrepair signature activity.
    opt_method : str
        Optimization method for variational inference. Either "ADVI" for mean-field inference or
        "FullRankADVI" for full-rank inference.
    seed : int
        Random seed for reproducibility.
    
    Attributes
    ----------
    model : pymc3.Model
        PyMC3 model instance.
    model_kwargs : dict
        Dictionary of parameters passed when constructing the model (e.g., hyperprior values).
    approx : pymc3.approximations.Approximation
        PyMC3 approximation object created via self.fit().
    run_id : str
        Unique identifier for the current run, used for saving checkpoint files and in wandb if enabled.
    """
    
    def __init__(self, dataset: DataSet, n_damage_sigs=None, n_misrepair_sigs=None,
                 alpha_bias=0.1, psi_bias=0.01, beta_bias=0.1, gamma_bias=0.01,
                 phi_obs = None, etaC_obs = None, etaT_obs = None, 
                 opt_method="ADVI", init_strategy="kmeans", init_signatures=None, seed=2021):
        
        super().__init__(dataset=dataset, opt_method=opt_method, init_strategy=init_strategy, init_signatures=init_signatures, seed=seed)
        # Check if n_damage_sigs and n_misrepair_sigs are None when obs are provided
        if phi_obs is None and n_damage_sigs is None:
            raise ValueError("n_damage_sigs must be provided when phi_obs is None")
        if phi_obs is not None:
            if n_damage_sigs is not None:
                warnings.warn("argument n_damage_sigs ({}) is ignored when phi_obs is None".format(n_damage_sigs))
            n_damage_sigs = phi_obs.shape[0]

        if (etaC_obs is not None and etaT_obs is None) or (etaC_obs is None and etaT_obs is not None):
            raise ValueError("etaC_obs and etaT_obs must be provided together.")
        if (etaC_obs is None) and n_misrepair_sigs is None:
            raise ValueError("n_misrepair_sigs must be provided when etaC_obs and etaT_obs are None")
        if etaC_obs is not None:
            if etaC_obs.shape[0] != etaT_obs.shape[0]:
                raise ValueError("etaC_obs.shape[0] ({}) must equal etaT_obs.shape[0] ({})".format(etaC_obs.shape[0], etaT_obs.shape[0]))
            if etaC_obs is not None:
                if n_misrepair_sigs is not None:
                    warnings.warn("argument n_misrepair_sigs ({}) is ignored when etaC_obs and etaT_obs are provided".format(n_misrepair_sigs))
                n_misrepair_sigs = etaC_obs.shape[0]

        assert n_damage_sigs is not None and n_misrepair_sigs is not None, "n_damage_sigs and n_misrepair_sigs must be provided"
        
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
        self._model_kwargs['phi_init'] = k_means(marginalize_for_phi(data), self.n_damage_sigs, init='k-means++',random_state=np.random.RandomState(self._rng.bit_generator))[0]
        eta = k_means(marginalize_for_eta(data).reshape(-1,6), self.n_misrepair_sigs, init='k-means++', random_state=np.random.RandomState(self._rng.bit_generator))[0].reshape(-1,2,3)
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
        """Compile a PyMC3 model for mutational signature analysis.
        
        Parameters 
        ----------
        n_damage_sigs : int
            Number of damage signatures to fit.
        n_misrepair_sigs : int
            Number of misrepair signatures to fit.
        alpha_bias : float or numpy.ndarray of shape (32,)
            Dirichlet concentration parameter for damage signature trinucleotide context priors.
        psi_bias : float or numpy.ndarray of shape (n_damage_sigs,)
            Dirichlet concentration parameter for damage signature activity priors.
        beta_bias : float or numpy.ndarray of shape (6,)
            Dirichlet concentration parameter for misrepair signature substitution type priors.
        gamma_bias : float or numpy.ndarray of shape (n_misrepair_sigs,)
            Dirichlet concentration parameter for misrepair signature activity priors.
        phi_init : numpy.ndarray of shape (n_damage_sigs, 32), optional
            Initial values for damage signatures.
        etaC_init : numpy.ndarray of shape (n_misrepair_sigs, 3), optional
            Initial values for C-context misrepair signatures.
        etaT_init : numpy.ndarray of shape (n_misrepair_sigs, 3), optional
            Initial values for T-context misrepair signatures.
        phi_obs : numpy.ndarray, optional
            Observed values for damage signatures (for partial fitting).
        etaC_obs : numpy.ndarray, optional
            Observed values for C-context misrepair signatures (for partial fitting).
        etaT_obs : numpy.ndarray, optional
            Observed values for T-context misrepair signatures (for partial fitting).
        """
        # latent dirichlet allocation with tandem signautres of damage and repair
        train = self.dataset.counts.to_numpy()
        
        S = train.shape[0]
        N = train.sum(1).reshape(S,1)
        J = n_damage_sigs
        K = n_misrepair_sigs
        
        with pm.Model() as self.model:
            
            data = pm.Data("data", train)
            phi = dirichlet('phi', a = np.ones(32) * alpha_bias, shape=(J, 32), testval = phi_init, observed = phi_obs)
            theta = dirichlet("theta", a = np.ones(J) * psi_bias, shape=(S, J))
            A = dirichlet("A", a = np.ones(K) * gamma_bias, shape = (S, J, K))
            # 4 is constant for ACGT
            beta = np.ones(4) * beta_bias
            etaC = dirichlet("etaC", a=beta[[0,2,3]], shape=(K, 3), testval = etaC_init, observed = etaC_obs)
            etaT = dirichlet("etaT", a=beta[[0,1,2]], shape=(K, 3), testval = etaT_init, observed = etaT_obs)
            eta = pm.Deterministic('eta', pm.math.stack([etaC, etaT], axis=1))

            B = pm.Deterministic("B", (pm.math.dot(theta, phi).reshape((S,2,16))[:,:,None,:] * \
                                    pm.math.dot(batched_dot(theta,A), eta.dimshuffle(1,0,2))[:,:,:,None]).reshape((S, -1)))
            
            # mutation counts
            pm.Multinomial('corpus', n = N, p = B, observed=data)

    
    def get_estimated_signatures(self, n_draws=1):
        """Extract damage and misrepair signatures from model posterior.
        
        Parameters
        ----------
        n_draws : int, default=1
            Number of posterior samples to draw.
            
        Returns
        -------
        tuple
            Tuple containing (phi, eta) where phi is damage signatures array 
            and eta is misrepair signatures array reshaped to (n_draws, n_sigs, 6).
        """
        self.check_is_fitted()
        hat = self.approx.sample(n_draws)
        return hat.phi, hat.eta.reshape(n_draws, -1, 6)

    def get_estimated_SignatureSet(self, n_draws=1):
        """Construct a SignatureSet object from model posterior samples.
        
        Parameters
        ----------
        n_draws : int, default=1
            Number of posterior samples to draw. If >1, samples are averaged in the signature space
            
        Returns
        -------
        SignatureSet
            SignatureSet constructed from damage and misrepair signatures.
        """
        self.check_is_fitted()
        phi, eta = self.get_estimated_signatures(n_draws)
        if n_draws > 1:
            warnings.warn("Signatures will be summarized as the mean of {} posterior samples".format(n_draws))
        damage = pd.DataFrame(phi.mean(0), columns=mut32)
        misrepair = pd.DataFrame(eta.mean(0).reshape(-1, 6), columns=mut6)
        damage.index = ['D'+str(n+1) for n in damage.index]
        misrepair.index = ['M'+str(n+1) for n in misrepair.index]
        return SignatureSet.from_damage_misrepair(damage, misrepair)

    def get_estimated_W(self, n_draws=1):
        """Extract signature activity tensor from model posterior.
        
        Parameters
        ----------
        n_draws : int, default=1
            Number of posterior samples to draw.
            
        Returns
        -------
        np.ndarray
            4D array of signature activities with shape (n_draws, n_samples, n_damage_sigs, n_misrepair_sigs).
        """
        self.check_is_fitted()

        hat = self.approx.sample(n_draws)

        theta = hat.theta
        A = hat.A

        # A has misrepair dimension last
        assert theta.shape == A.shape[:-1], "theta and A shape mismatch"

        W = (theta[:,:,:,None]*A)
        assert np.allclose(W.sum(-1), theta), "W does not sum to theta over misrepair axis"
        return W
    
    def get_estimated_activities(self, n_draws=1):
        """"""
        W = self.get_estimated_W(n_draws)
        theta = W.sum(-1)
        gamma = W.sum(-2)
        return theta, gamma

    def get_estimated_activities_DataFrame(self, n_draws=1):
        """Extract damage and misrepair signature activities as DataFrames.
        
        Parameters
        ----------
        n_draws : int, default=1
            Number of posterior samples to draw. If >1, samples are averaged in the activity space.
            
        Returns
        -------
        tuple
            Tuple of (theta, gamma) DataFrames where theta contains damage signature 
            activities and gamma contains misrepair signature activities.
        """
        self.check_is_fitted()
        W = self.get_estimated_W(n_draws)
        if n_draws > 1:
            warnings.warn(f"Signatures will be sumarized as the mean of {n_draws} posterior samples")
        theta = pd.DataFrame(W.sum(-1).mean(0), index = self.dataset.ids, columns = ["D"+str(i+1) for i in range(self.n_damage_sigs)])
        gamma = pd.DataFrame(W.sum(-2).mean(0), index = self.dataset.ids, columns = ["M"+str(i+1) for i in range(self.n_misrepair_sigs)])
        return theta, gamma

    def get_estimated_connections_DataFrame(self, n_draws=1):
        """Extract damage-misrepair signature connections as DataFrame.
        
        Parameters
        ----------
        n_draws : int, default=1
            Number of posterior samples to draw. If >1, samples are averaged in the connection space.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with damage-misrepair signature combinations as columns,
            with column names like 'D1_M1', 'D1_M2', etc.
        """
        self.check_is_fitted()
        W = self.get_estimated_W(n_draws)
        if n_draws > 1:
            warnings.warn("Signatures will be summarized as the mean of {} posterior samples".format(n_draws))
        W_df = pd.DataFrame(W.mean(0).reshape(self.dataset.n_samples, -1), index=self.dataset.ids)
        W_df = W_df.rename(columns=lambda x: 'D{}_M{}'.format(x//self.n_misrepair_sigs + 1, x%self.n_misrepair_sigs + 1))
        return W_df

    
class HierarchicalTandemLda(TandemLda):
    """Bayesian inference of mutational signatures and their activities using a Hierarchical Tandem LDA model.
    
    This class fits COSMIC-style mutational signatures using a Hierarchical Tandem LDA model,
    where damage signatures and misrepair signatures have separate sets of activities. 
    A tissue-type hierarchical prior is fitted over damage-misrepair signature associations
    to improve the interpretability of misrepair activity specificities.
    
    Parameters
    ----------
    dataset : DataSet
        Dataset containing mutation counts and sample annotations for fitting.
    n_damage_sigs : int
        Number of damage signatures to fit.
    n_misrepair_sigs : int
        Number of misrepair signatures to fit.
    type_col : str
        Name of the annotation column containing the tissue type of each sample.
    alpha_bias : float or numpy.ndarray of shape (32,)
        Dirichlet concentration parameter for damage signature trinucleotide context priors.
    psi_bias : float or numpy.ndarray of shape (n_damage_sigs,)
        Dirichlet concentration parameter for damage signature activity priors.
    beta_bias : float or numpy.ndarray of shape (6,)
        Dirichlet concentration parameter for misrepair signature substitution type priors.
    gamma_bias : float or numpy.ndarray of shape (n_misrepair_sigs,)
        Dirichlet concentration parameter for misrepair signature activity priors.
    opt_method : str
        Optimization method: "ADVI" for mean-field inference or "FullRankADVI" for full-rank inference.
    seed : int
        Random seed for reproducibility.
    
    Attributes
    ----------
    model : pymc3.Model
        PyMC3 model instance.
    model_kwargs : dict
        Dictionary of parameters used for constructing the model.
    approx : pymc3.approximations.Approximation
        PyMC3 approximation object created via self.fit().
    run_id : str
        Unique identifier for the current run, used for saving checkpoint files.
    """
    
    def __init__(self, dataset: DataSet, type_col: str,
                 n_damage_sigs=None, n_misrepair_sigs=None,
                 alpha_bias=0.1, psi_bias=0.01, beta_bias=0.1,  
                 phi_obs = None, etaC_obs = None, etaT_obs = None,
                 opt_method="ADVI", init_strategy="kmeans", init_signatures=None, seed=2021):
        
        
        super().__init__(dataset=dataset, n_damage_sigs = n_damage_sigs, n_misrepair_sigs = n_misrepair_sigs, 
                         alpha_bias = alpha_bias, psi_bias = psi_bias, beta_bias = beta_bias,
                         phi_obs = phi_obs, etaC_obs = etaC_obs, etaT_obs = etaT_obs,
                         opt_method=opt_method, init_strategy=init_strategy, init_signatures=init_signatures, seed=seed)
        self._model_kwargs.pop('gamma_bias')
        self.dataset.annotate_tissue_types(type_col) 

    def _build_model(self, n_damage_sigs, n_misrepair_sigs, alpha_bias, psi_bias, beta_bias, 
                     phi_init=None, etaC_init=None, etaT_init=None,
                     phi_obs=None, etaC_obs=None, etaT_obs=None):
        """Compile a PyMC3 model for mutation signature analysis.
        
        Parameters 
        ----------
        n_damage_sigs : int
            Number of damage signatures to fit.
        n_misrepair_sigs : int
            Number of misrepair signatures to fit.
        alpha_bias : float or numpy.ndarray of shape (32,)
            Dirichlet concentration parameter on (0, inf) for damage signature trinucleotide context priors.
        psi_bias : float or numpy.ndarray of shape (n_damage_sigs,)
            Dirichlet concentration parameter on (0, inf) for damage signature activity priors.
        beta_bias : float or numpy.ndarray of shape (6,)
            Dirichlet concentration parameter on (0, inf) for misrepair signature substitution type priors.
        phi_init : numpy.ndarray of shape (n_damage_sigs, 32), optional
            Initial values for damage signatures.
        etaC_init : numpy.ndarray of shape (n_misrepair_sigs, 3), optional
            Initial values for C-context misrepair signatures.
        etaT_init : numpy.ndarray of shape (n_misrepair_sigs, 3), optional
            Initial values for T-context misrepair signatures.
        phi_obs : numpy.ndarray of shape (n_damage_sigs, 32), optional
            Observed values for damage signatures.
        etaC_obs : numpy.ndarray of shape (n_misrepair_sigs, 3), optional
            Observed values for C-context misrepair signatures.
        etaT_obs : numpy.ndarray of shape (n_misrepair_sigs, 3), optional
            Observed values for T-context misrepair signatures.

        Returns
        -------
        pymc3.Model
            Compiled PyMC3 model for mutation signature analysis.
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

    def get_estimated_W(self, n_draws=1):
        """Extract signature activity tensor from model posterior.
        
        Parameters
        ----------
        n_draws : int, default=1
            Number of posterior samples to draw.
            
        Returns
        -------
        np.ndarray
            4D array of signature activities with shape (n_draws, n_samples, n_damage_sigs, n_misrepair_sigs).
        """
        self.check_is_fitted()

        hat = self.approx.sample(n_draws)

        theta = hat.theta
        A = hat.A
        A = np.moveaxis(A,1,2)

        # A has misrepair dimension last
        assert theta.shape == A.shape[:-1], "theta and A shape mismatch"

        W = (theta[:,:,:,None]*A)
        assert np.allclose(W.sum(-1), theta), "W does not sum to theta over misrepair axis"
        return W