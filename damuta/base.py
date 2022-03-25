import numpy as np
import pandas as pd
import pymc3 as pm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .constants import *
from .utils import *

__all__ = ['Damuta', 'DataSet', 'SignatureSet']

_opt_methods = {"ADVI": pm.ADVI, "FullRankAdvi": pm.FullRankADVI}

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
    >>> pcawg.S
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
    def nsamples(self) -> int:
        """Number of samples in dataset"""
        return self.counts.shape[0]
    
    @property
    def ids(self) -> list:
        """List sample ids in dataset"""
        return self.counts.index.to_list()
    
    def annotate_tissue_types(self, type_col) -> np.array:
        """Set a specified column of annotation as the sample tissue type
        
        Tissue type information is used by hirearchical models to create tissue-type prior.
        See class:`HierarchicalTendemLda` for mor info. 
        """
        if self.annotation is None:
            raise ValueError('self.annotation must be first provided.')
        self.tissue_types = pd.Categorical(self.annotation['type_col'])
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
        
    @property
    def nsigs(self) -> int:
        """Number of signatures in dataset"""
        return self.signatures.shape[0]
    
    @property
    def damage_signatures(self) -> pd.DataFrame:
        """Damage signatures 
        
        Damage signatures represent the distribution of mutations over 32 trinucleotide contexts. 
        They are computed by marginalizing over substitution classes. 
        """
        phi = get_phi(self.signatures.to_numpy())
        return pd.DataFrame(phi, index = self.signatures.index, columns=mut32)
        
                
    @property
    def misrepair_signatures(self) -> pd.DataFrame:
        """Misrepair signatures 
        
        Misrepair signatures represent the distribution of mutations over 6 substitution types. 
        They are computed by marginalizing over trinucleotide context classes. 
        """
        eta = get_eta(self.signatures.to_numpy())
        return pd.DataFrame(eta, index = self.signatures.index, columns=mut6)
    
    def summarize_separation(self) -> pd.DataFrame:
        """Summary statistics of pair-wise cosine distances for signautres, 
        damage signatures, and misrepair signatures.
        
        """
        
        seps = {'Signature separation': cosine_similarity(self.signatures)[np.triu_indices(self.nsigs, k=1)],
                'Damage signature separation': cosine_similarity(self.damage_signatures)[np.triu_indices(self.nsigs, k=1)],
                'Misrepair signature separation': cosine_similarity(self.misrepair_signatures)[np.triu_indices(self.nsigs, k=1)]
               }
        
        return pd.DataFrame.from_dict(seps).describe()
    
class Damuta(ABC):
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
    seed : int
        Random seed
    Attributes
    ----------
    model: pymc3.model.Model object
        pymc3 model instance
    model_kwargs: dict
        dict of parameters to pass when constructing model (ex. hyperprior values)
    approx: pymc3.variational.approximations object
        pymc3 approximation object. Created via self.fit()
    run_id: str
        Unique label used to identify run. Used when saving checkpoint files, drawn from wandb run if wandb is enabled.
    """

    def __init__(self, dataset: DataSet, opt_method: str, seed=9595):
        if not isinstance(dataset, DataSet):
            raise TypeError('Learner instance must be initialized with a DataSet object')

        if not opt_method in _opt_methods.keys():
            raise TypeError(f'Optimization method should be one of {list(_opt_methods.keys())}')
        
        self.dataset = dataset
        self.opt_method = opt_method
        self.seed = seed
        self.model = None
        self.model_kwargs = None
        self.approx = None
        self.run_id = None
        
        
        # hidden attributes
        self._opt = _opt_methods[self.opt_method]
        self._trace = None
        self._hat = None
        self._rng = np.random.default_rng(self.seed)

    ################################################################################
    # Model building and fitting
    ################################################################################

    @abstractmethod
    def build_model(self, *args, **kwargs):
        """Defined by subclass
        """
        pass
    
    def fit(self, n_iter, callbacks=None, **pymc3_kwargs):
        """Fit model to the dataset specified by self.dataset
        
        Parameters 
        ----------
        n_iter: int
            Number of iterations 
        callbacks: list
            List of callbacks to pass to pymc3.fit() while fitting
        **pymc3_kwargs:
            More parameters to pass to pymc3.fit()
            
        Returns
        -------
        self : :class:`Lda`
        """
        
        self.build_model(**self.model_kwargs)
        
        with self.model:
            self._trace = self.opt.fit(n=n_iter, random_seed = self.seed, **pymc3_kwargs)
        
        self.approx = self._trace.approx
        
        return self
    
    #@abstractmethod
    #def predict_activites(self, new_data, *args, **kwargs):
    #    """Defined by subclass
    #    """
    #    pass
    

    #def model_to_gv(self, *args, **kwargs):
    #    """Defined by subclass
    #    """
    #    pass

    
    # metrics
    @abstractmethod
    def BOR(self, *args, **kwargs):
        """Defined by subclass
        """
        pass
    
    def ALP(self, n_samples = 20):
        """Average log probability per mutation 
        """
        if self.approx is None:
            warnings.warn("self.approx is None... Fit the model first!", ValueError)
        
        B = self.approx.sample(n_samples).B.mean(0)
        return alp_B(self.dataset.counts.to_numpy(), B)

    
    def LAP(self, n_samples = 20, *args, **kwargs):
        """Log average data likelihood
        """
        if self.approx is None:
            warnings.warn("self.trace is None... Fit the model first!", ValueError)
        
        B = self.trace.approx.sample(n_samples).B.mean(0)
        return alp_B(self.dataset.counts.to_numpy(), B)
    
    def reconstruction_err(self, *args, **kwargs):
        """Defined by subclass
        """
        pass
    

