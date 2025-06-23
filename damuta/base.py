import numpy as np
import pandas as pd
import pymc3 as pm
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .constants import *
from .utils import *

__all__ = ['Damuta', 'DataSet', 'SignatureSet']

_opt_methods = {"ADVI": pm.ADVI, "FullRankADVI": pm.FullRankADVI}
_init_strats = ['uniform', 'kmeans', 'from_sigs']

@dataclass
class DataSet:
    """Container for tabular data, allowing simple access to a mutation data set and corresponding annotation for each sample.
    
    :class:`DataSet` is instatiated from a pandas dataframe of mutation counts, and (optionally) a pandas dataframe of the
    same size of sample annotations. The dataframe index is taken as sample ids. All samples that appear in counts should 
    also appear in annotation, and vice versa. Mutation types are expect to be in COSMIC format (ex. A[C>A]A). 
    
    Parameters
    ----------
    counts: pd.DataFrame
        Nx96 dataframe of mutation counts, one sample per row. Index is assumed to be sample ids.
    annotation: pd.DataFrame
        NxF dataframe of meta-data features to annotate samples with. Index is assumed to be sample ids.

    Examples
    --------
    >>> import pandas as pd
    >>> counts = pd.read_csv('tests/test_data/pcawg_counts.csv', index_col = 0, header = 0)
    >>> annotation = pd.read_csv('tests/test_data/pcawg_cancer_types.csv', index_col = 0, header = 0)
    >>> pcawg = DataSet(counts, annotation)
    >>> pcawg.nsamples
    2778
    """

    counts: pd.DataFrame
    annotation: pd.DataFrame = None

    def __post_init__(self):
        if self.counts is not None:
            assert self.counts.ndim == 2, f'Expected counts.ndim==2. Got {self.counts.ndim}'
            assert self.counts.shape[1] == 96, f'Expected 96 mutation types, got {self.counts.shape[1]}'
            assert all(self.counts.columns.isin(mut96)), 'Unexpected mutation type. Check the counts.columns are in COSMIC mutation type format (ex. A[C>A]A). See COSMIC database for more.'
            # reorder columns if necessary
            self.counts = self.counts[mut96]
            
        if self.annotation is not None:
            # check the counts and annotation match
            assert self.annotation.shape[0] == self.counts.shape[0], f"Shape mismatch. Expected self.annotation.shape[0] == self.counts.shape[0], got {self.annotation.shape[0]}, {self.counts.shape[0]}"
            assert self.annotation.index.isin(self.counts.index).all() and self.counts.index.isin(self.annotation.index).all(), "Counts and annotation indices must match"

    @property
    def n_samples(self) -> int:
        """Number of samples in dataset"""
        return self.counts.shape[0]
    
    @property
    def ids(self) -> list:
        """List sample ids in dataset"""
        return self.counts.index.to_list()
    
    def annotate_tissue_types(self, type_col) -> np.array:
        """Set a specified column of annotation as the sample tissue type
        
        Tissue type information is used by hirearchical models to create tissue-type prior.
        See class:`HierarchicalTendemLda` for more details. 
        """
        if self.annotation is None:
            raise ValueError('Dataset annotation must be provided.')
        assert type_col in self.annotation.columns, f"{type_col} not found in annotation columns. Check spelling?"
        self.tissue_types = pd.Categorical(self.annotation[type_col])
        self.type_codes = self.tissue_types.codes
  
@dataclass
class SignatureSet:
    """Container for tabular data, allowing simple access to a set of mutational signature definitions. 
    
    Parameters
    ----------
    signatures: pd.DataFrame
        Nx96 dataframe of signautre definitions, one signature per row. Rows must sum to 1.
        
    Examples
    ----------
    """
    
    signatures: pd.DataFrame
    
    def __post_init__(self):
        # check for shape, valid signautre definitions
        assert self.signatures.shape[1] == 96, f"Expected 96 mutation types, got {self.signatures.shape[1]}"
        assert np.allclose(self.signatures.sum(1),1), "All signature definitions must sum to 1"
        assert self.signatures.columns.isin(mut96).all(), 'Check signature column names'
        self.signatures = self.signatures[mut96]
        _phi = marginalize_for_phi(self.signatures.to_numpy())
        self.damage_signatures = pd.DataFrame(_phi, index = self.signatures.index, columns=mut32)
        _eta = marginalize_for_eta(self.signatures.to_numpy())
        self.misrepair_signatures = pd.DataFrame(_eta, index = self.signatures.index, columns=mut6)
    
    @classmethod
    def from_damage_misrepair(cls, damage_signatures: pd.DataFrame, misrepair_signatures: pd.DataFrame):
        assert damage_signatures.columns.isin(mut32).all(), 'Check damage_signatures column names'
        assert misrepair_signatures.columns.isin(mut6).all(), 'Check misrepair_signatures column names'
        phi = damage_signatures.copy()[mut32]
        eta = misrepair_signatures.copy()[mut6]
        cross_prod = get_tau(phi.to_numpy(), eta.to_numpy().reshape(-1,2,3))
        c = cls(pd.DataFrame(cross_prod, columns = mut96,
                             index = [str(d) + '_' + str(m) for d in phi.index for m in eta.index]))
        c.damage_signatures = phi
        c.misrepair_signatures = eta
        return c
    
    @property
    def index(self) -> int:
        """Names of signatures in dataset"""
        return self.signatures.index
        
    @property
    def n_sigs(self) -> int:
        """Number of signatures in dataset"""
        return self.signatures.shape[0]
    
    @property
    def n_damage_sigs(self) -> pd.DataFrame:
        """Number of damage signatures in dataset
        
        Damage signatures represent the distribution of mutations over 32 trinucleotide contexts. 
        They are computed by marginalizing over substitution classes. 
        """
        return self.damage_signatures.shape[0]
        
                
    @property
    def n_misrepair_sigs(self) -> pd.DataFrame:
        """Number of misrepair signatures in dataset
        
        Misrepair signatures represent the distribution of mutations over 6 substitution types. 
        They are computed by marginalizing over trinucleotide context classes. 
        """
        return self.misrepair_signatures.shape[0]
    
    def summarize_separation(self) -> pd.DataFrame:
        """Summary statistics of pair-wise cosine distances for signatures, 
        damage signatures, and misrepair signatures.
        """
        
        # Calculate each array separately
        sig_seps = cosine_similarity(self.signatures)[np.triu_indices(self.n_sigs, k=1)]
        damage_seps = cosine_similarity(self.damage_signatures)[np.triu_indices(self.n_damage_sigs, k=1)]
        misrepair_seps = cosine_similarity(self.misrepair_signatures)[np.triu_indices(self.n_misrepair_sigs, k=1)]
        
        # Create separate DataFrames for each
        results = pd.DataFrame()
        
        results['Mutational signature similarity'] = pd.Series(sig_seps).describe()
        results['Damage signature similarity'] = pd.Series(damage_seps).describe()
        results['Misrepair signature similarity'] = pd.Series(misrepair_seps).describe()
        
        return results
    
class Model(ABC):
    """
    Bayesian inference of mutational signautres and their activities.
    
    The Damuta class acts as a central interface for several types of latent models. Each subclass defines at least `build_model`, 
    `fit`, `predict_activities`, `model_to_gv` and metrics such as `LAP`, `ALP`, and `BOR` in addition to subclass-specific methods.
    
    Parameters
    ----------
    dataset : DataSet
        Data for fitting.
    opt_method: str 
        one of "ADVI" for mean field inference, or "FullRankADVI" for full rank inference.
    init_strategy: str
        one of "uniform", "kmeans", "from_sigs"
    init_signatures: SignatureSet
        set of signatures to initialize from, required if init_strategy is set to "from_sigs"
    seed : int
        Random seed
    
    Attributes
    ----------
    model: pymc3.model.Model object
        pymc3 model instance
    approx: pymc3.variational.approximations object
        pymc3 approximation object. Created via self.fit()
     """

    def __init__(self, dataset: DataSet, opt_method: str, init_strategy: str, init_signatures: SignatureSet, seed: int):
        
        if not isinstance(dataset, DataSet):
            raise TypeError('Learner instance must be initialized with a DataSet object')

        if not opt_method in _opt_methods.keys():
            raise TypeError(f'Optimization method should be one of {list(_opt_methods.keys())}')
        assert init_strategy in _init_strats, f'self.init_strategy should be one of {_init_strats}'
        
        if init_strategy == 'from_sigs':
            assert init_signatures is not None, 'init_strategy "from_sigs" requires a signature set to be passed.'
        if init_signatures is not None and init_strategy != 'from_sigs': 
            warnings.warn('signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.')
        
        self.dataset = dataset
        self.opt_method = opt_method
        self.init_strategy = init_strategy
        self.init_signatures = init_signatures
        self.seed = seed
        self.model = None
        self.approx = None
        self.fitted_ = False

        # hidden attributes
        self._model_kwargs = None
        self._opt = _opt_methods[self.opt_method]
        self._trace = None
        self._rng = np.random.default_rng(self.seed)
        
        # set seed
        np.random.seed(self.seed)
        pm.set_tt_rng(self.seed)

    ################################################################################
    # Model building and fitting
    ################################################################################

    @abstractmethod
    def _build_model(self, *args, **kwargs):
        """Build the pymc3 model 
        """
        pass
    
    @abstractmethod
    def _init_uniform(self):
        """Initialize signatures uniformly 
        Defined by subclass
        """
        pass
    
    @abstractmethod
    def _init_kmeans(self):
        """Initialize signatures via kmeans on the data
        Defined by subclass
        """
        pass
    
    @abstractmethod
    def _init_from_sigs(self):
        """Initialize signatures from a supplied SignatureSet
        Defined by subclass
        """
        pass

    def _validate_init_sigs(self):
        """ Check that sigs used for initialization are valid
        """
        if "tau_init" in self._model_kwargs.keys():
            assert (self._model_kwargs["tau_init"] is None) or np.allclose(self._model_kwargs["tau_init"].sum(1), 1)
        if "phi_init" in self._model_kwargs.keys():
            assert (self._model_kwargs["phi_init"] is None) or np.allclose(self._model_kwargs["phi_init"].sum(1), 1)
        # eta should be kxpxm
        if "etaC_init" in self._model_kwargs.keys():
            assert (self._model_kwargs["etaC_init"] is None) or np.allclose(self._model_kwargs["etaC_init"].sum(1), 1)
        if "etaT_init" in self._model_kwargs.keys():
            assert (self._model_kwargs["etaT_init"] is None) or np.allclose(self._model_kwargs["etaT_init"].sum(1), 1)
    
    def _initialize_signatures(self):
        """Initialize signatures using selected initialization strategy
        """
        if self.init_strategy == "uniform":
            self._init_uniform()
            
        if self.init_strategy == "kmeans":
            self._init_kmeans()
            
        if self.init_strategy == "from_sigs":
            assert self.init_signatures is not None, "init_signatures required for init_strategy 'from_sigs'"
            self._init_from_sigs()
        
        self._validate_init_sigs()

    def check_is_fitted(self):
        """Check if the model has been fitted.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.fitted_:
            raise ValueError("This model instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
    

    def fit(self, n, **pymc3_kwargs):
        """Fit model to the dataset specified by self.dataset
        
        Parameters 
        ----------
        n: int
            Number of iterations 
        **pymc3_kwargs:
            More parameters to pass to pymc3.fit() (ex. callbacks)
            
        Returns
        -------
        self: :class:`Lda`
        """
        
        self._initialize_signatures()
        self._build_model(**self._model_kwargs)
        
        with self.model:
            self._trace = self._opt(random_seed = self.seed)
            self._trace.fit(n=n, **pymc3_kwargs)
        
        self.approx = self._trace.approx
        #self._hat = self.approx.sample(1)
        self.fitted_ = True
        
        return self
    
    
    ################################################################################
    # Metrics
    ################################################################################
    
    def ALP(self, n_samples = 20):
        """Average log probability per mutation 
        """
        if self.approx is None: warnings.warn("self.approx is None... Fit the model first!", ValueError)
        
        B = self.approx.sample(n_samples).B.mean(0)
        return alp_B(self.dataset.counts.to_numpy(), B)

    
    def LAP(self, n_samples = 20):
        """Log average data likelihood (bayesian version of reconstruction error)
        """
        if self.approx is None: warnings.warn("self.trace is None... Fit the model first!", ValueError)
        
        B = self.approx.sample(n_samples).B
        return lap_B(self.dataset.counts.to_numpy(), B)
    
    def reconstruction_err(self, *args, **kwargs):
        """Defined by subclass
        """
        pass
    
    ################################################################################
    # Posterior sampling
    ################################################################################

    @abstractmethod
    def get_estimated_signatures(self, n_draws=1):
        """Extract signatures from model posterior - implemented by subclass"""
        pass

    @abstractmethod 
    def get_estimated_SignatureSet(self, n_draws=1):
        """Construct SignatureSet from posterior - implemented by subclass"""
        pass

    @abstractmethod
    def get_estimated_activities_DataFrame(self, n_draws=1):
        """Extract activities as DataFrame - implemented by subclass"""
        pass