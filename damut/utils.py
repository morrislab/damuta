import numpy as np
import pymc3 as pm
import pandas as pd
import pickle
import wandb
import yaml
import warnings
from sklearn.cluster import k_means
from scipy.special import softmax, logsumexp, loggamma
from sklearn.metrics.pairwise import cosine_similarity

# constants
C=32
M=3
P=2

# labels
idx96 = pd.MultiIndex.from_tuples([
            ('C>A', 'ACA'), ('C>A', 'ACC'), ('C>A', 'ACG'), ('C>A', 'ACT'), 
            ('C>A', 'CCA'), ('C>A', 'CCC'), ('C>A', 'CCG'), ('C>A', 'CCT'), 
            ('C>A', 'GCA'), ('C>A', 'GCC'), ('C>A', 'GCG'), ('C>A', 'GCT'), 
            ('C>A', 'TCA'), ('C>A', 'TCC'), ('C>A', 'TCG'), ('C>A', 'TCT'), 
            ('C>G', 'ACA'), ('C>G', 'ACC'), ('C>G', 'ACG'), ('C>G', 'ACT'), 
            ('C>G', 'CCA'), ('C>G', 'CCC'), ('C>G', 'CCG'), ('C>G', 'CCT'), 
            ('C>G', 'GCA'), ('C>G', 'GCC'), ('C>G', 'GCG'), ('C>G', 'GCT'), 
            ('C>G', 'TCA'), ('C>G', 'TCC'), ('C>G', 'TCG'), ('C>G', 'TCT'), 
            ('C>T', 'ACA'), ('C>T', 'ACC'), ('C>T', 'ACG'), ('C>T', 'ACT'), 
            ('C>T', 'CCA'), ('C>T', 'CCC'), ('C>T', 'CCG'), ('C>T', 'CCT'), 
            ('C>T', 'GCA'), ('C>T', 'GCC'), ('C>T', 'GCG'), ('C>T', 'GCT'), 
            ('C>T', 'TCA'), ('C>T', 'TCC'), ('C>T', 'TCG'), ('C>T', 'TCT'), 
            ('T>A', 'ATA'), ('T>A', 'ATC'), ('T>A', 'ATG'), ('T>A', 'ATT'), 
            ('T>A', 'CTA'), ('T>A', 'CTC'), ('T>A', 'CTG'), ('T>A', 'CTT'), 
            ('T>A', 'GTA'), ('T>A', 'GTC'), ('T>A', 'GTG'), ('T>A', 'GTT'), 
            ('T>A', 'TTA'), ('T>A', 'TTC'), ('T>A', 'TTG'), ('T>A', 'TTT'), 
            ('T>C', 'ATA'), ('T>C', 'ATC'), ('T>C', 'ATG'), ('T>C', 'ATT'), 
            ('T>C', 'CTA'), ('T>C', 'CTC'), ('T>C', 'CTG'), ('T>C', 'CTT'), 
            ('T>C', 'GTA'), ('T>C', 'GTC'), ('T>C', 'GTG'), ('T>C', 'GTT'), 
            ('T>C', 'TTA'), ('T>C', 'TTC'), ('T>C', 'TTG'), ('T>C', 'TTT'), 
            ('T>G', 'ATA'), ('T>G', 'ATC'), ('T>G', 'ATG'), ('T>G', 'ATT'), 
            ('T>G', 'CTA'), ('T>G', 'CTC'), ('T>G', 'CTG'), ('T>G', 'CTT'), 
            ('T>G', 'GTA'), ('T>G', 'GTC'), ('T>G', 'GTG'), ('T>G', 'GTT'), 
            ('T>G', 'TTA'), ('T>G', 'TTC'), ('T>G', 'TTG'), ('T>G', 'TTT')],
            names=['Type', 'Subtype'])
    
mut96 = ['A[C>A]A', 'A[C>A]C', 'A[C>A]G', 'A[C>A]T', 'C[C>A]A', 'C[C>A]C', 'C[C>A]G', 'C[C>A]T', 
         'G[C>A]A', 'G[C>A]C', 'G[C>A]G', 'G[C>A]T', 'T[C>A]A', 'T[C>A]C', 'T[C>A]G', 'T[C>A]T', 
         'A[C>G]A', 'A[C>G]C', 'A[C>G]G', 'A[C>G]T', 'C[C>G]A', 'C[C>G]C', 'C[C>G]G', 'C[C>G]T', 
         'G[C>G]A', 'G[C>G]C', 'G[C>G]G', 'G[C>G]T', 'T[C>G]A', 'T[C>G]C', 'T[C>G]G', 'T[C>G]T', 
         'A[C>T]A', 'A[C>T]C', 'A[C>T]G', 'A[C>T]T', 'C[C>T]A', 'C[C>T]C', 'C[C>T]G', 'C[C>T]T', 
         'G[C>T]A', 'G[C>T]C', 'G[C>T]G', 'G[C>T]T', 'T[C>T]A', 'T[C>T]C', 'T[C>T]G', 'T[C>T]T', 
         'A[T>A]A', 'A[T>A]C', 'A[T>A]G', 'A[T>A]T', 'C[T>A]A', 'C[T>A]C', 'C[T>A]G', 'C[T>A]T', 
         'G[T>A]A', 'G[T>A]C', 'G[T>A]G', 'G[T>A]T', 'T[T>A]A', 'T[T>A]C', 'T[T>A]G', 'T[T>A]T', 
         'A[T>C]A', 'A[T>C]C', 'A[T>C]G', 'A[T>C]T', 'C[T>C]A', 'C[T>C]C', 'C[T>C]G', 'C[T>C]T', 
         'G[T>C]A', 'G[T>C]C', 'G[T>C]G', 'G[T>C]T', 'T[T>C]A', 'T[T>C]C', 'T[T>C]G', 'T[T>C]T', 
         'A[T>G]A', 'A[T>G]C', 'A[T>G]G', 'A[T>G]T', 'C[T>G]A', 'C[T>G]C', 'C[T>G]G', 'C[T>G]T', 
         'G[T>G]A', 'G[T>G]C', 'G[T>G]G', 'G[T>G]T', 'T[T>G]A', 'T[T>G]C', 'T[T>G]G', 'T[T>G]T']

mut32 = ['ACA', 'ACC', 'ACG', 'ACT', 'CCA', 'CCC', 'CCG', 'CCT', 
         'GCA', 'GCC', 'GCG', 'GCT', 'TCA', 'TCC', 'TCG', 'TCT', 
         'ATA', 'ATC', 'ATG', 'ATT', 'CTA', 'CTC', 'CTG', 'CTT', 
         'GTA', 'GTC', 'GTG', 'GTT', 'TTA', 'TTC', 'TTG', 'TTT']

mut16 = ['A_A', 'A_C', 'A_G', 'A_T', 'C_A', 'C_C', 'C_G', 'C_T', 
         'G_A', 'G_C', 'G_G', 'G_T', 'T_A', 'T_C', 'T_G', 'T_T']

mut6 = ['C>A','C>G','C>T','T>A','T>C','T>G']


def load_config(config_fp):

    # load the yaml file 
    with open(config_fp, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Loaded configuration file {config_fp}")

    # remove any parameters not applicable to selected data source
    ds = config.pop('dataset')
    ds[ds['dataset_sel']].update({'dataset_sel': ds['dataset_sel']})

    # update dataset args to subsetted list
    config.update({'dataset': ds[ds['dataset_sel']]})
    
    # handle seeding
    config['dataset'].update({'data_rng': np.random.default_rng(config['dataset']['data_seed'])})
    config['model'].update({'model_rng': np.random.default_rng(config['model']['model_seed'])})
    
    return config['dataset'], config['model'], config['pymc3']
    
def detect_naming_style(fp):

    # first column must have type at minimum
    df = pd.read_csv(fp, index_col=0, sep = None, engine = 'python')
    naming_style = 'unrecognized'
    
    # check if index is type style
    if df.index.isin(mut96).any(): naming_style = 'type'

    # check if index is type/subtype style
    else:
        df = df.reset_index()
        df = df.set_index(list(df.columns[0:2]))
        if df.index.isin(idx96).any(): naming_style = 'type/subtype'

    assert naming_style == 'type' or naming_style == 'type/subtype', \
            'Mutation type naming style could not be identified.\n'\
            '\tExpected either two column type/subtype (ex. C>A,ACA) or\n'\
            '\tsingle column type (ex A[C>A]A). See examples at COSMIC database.'
    
    return naming_style

def load_sigs(fp):

    naming_style = detect_naming_style(fp)

    if naming_style == 'type':
        # read in sigs
        sigs = pd.read_csv(fp, index_col = 0, sep = None, engine = 'python').reindex(mut96)
        # sanity check for matching mut96, should have no NA 
        sel = (~sigs.isnull()).all(axis = 1)
        assert sel.all(), f'invalid signature definitions: null entry for types {list(sigs.index[~sel])}' 
        # convert to pcawg convention
        sigs = sigs.set_index(idx96)
        
    elif naming_style == 'type/subtype':
        # read in sigs
        sigs = pd.read_csv(fp, index_col = (0,1), header=0).reindex(idx96)
        # sanity check for idx, should have no NA
        sel = (~sigs.isnull()).all(axis = 1)
        assert sel.all(), f'invalid signature definitions: null entry for types {list(sigs.index[~sel])}' 
    
    # check colsums are 1
    sel = np.isclose(sigs.sum(axis=0), 1)
    assert sel.all(), f'invalid signature definitions: does not sum to 1 in columns {list(sigs.columns[~sel])}' 

    assert all(sigs.index == idx96) or all(sigs.index == mut96), 'signature defintions failed to be read correctly'
    
    # force Jx96 and mut96 convention
    sigs = sigs.T
    sigs.columns = mut96
    
    return sigs

def load_counts(counts_fp):
    counts = pd.read_csv(counts_fp, index_col = 0, header = 0)[mut96]
    assert counts.ndim == 2, 'Mutation counts failed to load. Check column names are mutation type (ex. A[C>A]A). See COSMIC database for more.'
    assert counts.shape[1] == 96, f'Expected 96 mutation types, got {counts.shape[1]}'
    return counts

def subset_samples(dataset, annotation, annotation_subset, sel_idx = 0):
    # subset sample ids by matching to annotation_subset
    # expect annotation_subset to be pd dataframe with ids as index

    if annotation_subset is None:
        return dataset

    # stop string being auto cast to list
    if type(annotation_subset) == str:
        annotation_subset = [annotation_subset]
    
    if annotation.ndim > 2:
        warnings.warn(f"More than one annotation is available per sample, selection index {sel_idx}", UserWarning)
        
    # annotation ids should match sample ids
    assert dataset.index.isin(annotation.index).any(), 'No sample ID matches found in dataset for the provided annotation'

    # reoder annotation (with gaps) to match dataset
    annotation = annotation.reindex(dataset.index).dropna()

    # partial matches allowed
    sel = np.fromiter((map(any, zip(*[annotation[annotation.columns[sel_idx]].str.contains(x) for x in annotation_subset] ))), dtype = bool)
        
    # type should appear in the type column of the lookup 
    assert sel.any(), 'Cancer type subsetting yielded no selection. Check keywords?'

    dataset = dataset.loc[annotation.index[sel]]
    return dataset

def save_checkpoint(fp, model, trace, dataset_args, model_args, pymc3_args): 
    with open(f'{fp}', 'wb') as buff:
        pickle.dump({'model': model, 'trace': trace, 'dataset_args': dataset_args, 
                     'model_args': model_args, 'pymc3_args': pymc3_args}, buff)
        
def load_checkpoint(fn):
    with open(fn, 'rb') as buff:
        data = pickle.load(buff)
        return data['model'], data['trace'], data['dataset_args'], data['model_args'], data['pymc3_args'] 

def split_by_count(data, fraction, rng):
    # assumes data is a pandas df

    frac1 = data.apply(lambda r: rng.choice( np.repeat(np.arange(96), r), int(fraction*r.sum()), replace = False) , axis = 1)
    frac1 = pd.DataFrame(frac1.apply(lambda r: np.histogram(r, bins=96, range=(0, 96))[0]).to_list(), 
                         index = data.index, columns = data.columns)
    #frac1 = (data * fraction).astype(int)
    #frac1 = frac1.apply(lambda r: np.histogram(r, bins=96, range=(0, 96))[0], axis = 1, result_type='expand')
    #frac1.columns = data.columns
    frac2 = data - frac1
    assert all(frac2 >= 0) and all(frac1 >= 0)
    assert np.all(data == frac1 + frac2), 'Splitting failed'
    return frac1, frac2

def split_by_S(data, fraction, rng):
    # assumes data is a pandas df with an index
    frac1 = data.sample(frac=fraction, random_state=np.random.RandomState(rng.bit_generator))
    frac2 = data.drop(frac1.index)
    return frac1, frac2

def split_data(counts, S_frac = 0.9, c_frac = 0.8, rng=np.random.default_rng()):
    # get train/val/test split
    trn, tst = split_by_S(counts, S_frac, rng)
    trn, val = split_by_count(trn, c_frac, rng)
    tst1, tst2 = split_by_count(tst, c_frac, rng)
    return trn, val, tst1, tst2

def get_tau(phi, eta):
    assert len(phi.shape) == 2 and len(eta.shape) == 3
    tau =  np.einsum('jpc,kpm->jkpmc', phi.reshape((-1,2,16)), eta).reshape((-1,96))
    return tau

def get_phi(sigs):
    # for each signature in sigs, get the corresponding phi
    wrapped = sigs.reshape(-1, 2, 3, 16)
    phi = wrapped.sum(2).reshape(-1,32)
    return phi

def get_eta(sigs):
    # for each signature in sigs, get the corresponding eta (kxpxm)
    wrapped = sigs.reshape(-1, 6, 16)
    eta = wrapped.sum(2).reshape(-1,3)
    # notrmalize such that etaC and etaT sum to 1 respectively.
    eta = (eta/eta.sum(1)[:,None]).reshape(-1,P,M)
    return eta

def flatten_eta(eta): # eta pkm -> kc
    warnings.warn('Eta no longer constructed as pkm - use reshape instead', DeprecationWarning)
    return np.moveaxis(eta,0,1).reshape(-1, 6)

def alr(x, e=1e-12):
    '''
    additive log ratio
    x is a NxD matrix
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr(x)
    array([ 1.09861229,  1.38629436,  0.69314718])
    '''
    # add small value for stability in log
    x = x + e
    return (np.log(x) - np.log(x[...,-1]).reshape(-1,1))[:,0:-1]

def alr_inv(y):
    '''
    inverse alr transform
    y is a Nx(D-1) matrix
    >>> x = np.array([.1, .3, .4, .2])
    >>> alr_inv(alr(x))
    array([ 0.1,  0.3,  0.4,  0.2])
    '''
    if y.ndim == 1: y = y.reshape(1,-1)
    return softmax(np.hstack([y, np.zeros((y.shape[0], 1)) ]), axis = 1)

def kmeans_alr(data, nsig, rng=np.random.default_rng()):
    #https://github.com/scikit-learn/scikit-learn/issues/16988#issuecomment-817375063
    km = k_means(alr(data), nsig, random_state=np.random.RandomState(rng.bit_generator))
    return alr_inv(km[0])

def alp_B(data, B):
    return (data * np.log(B)).sum() / data.sum()

def mult_ll(x, p):
    # x and p should both be same dimensions; Sx96
    return loggamma(x.sum(1) + 1) - loggamma(x+1).sum(1) + (x * np.log(p)).sum(1)

def lap_B(data, Bs):
    # Bs should be shape DxSx96 where D is the number of posterior samples
    # use logsumexp for stability
    assert Bs.ndim == 3, 'expected multiple trials for B'
    return logsumexp(np.vstack([mult_ll(data, B) for B in Bs]).sum(1)) - np.log(Bs.shape[0])

def profile_sigs(sigs, refsigs, thresh = 0.9, refidx = None):
    # return mapping of sigs to similar refsigs
    
    if isinstance(refsigs, pd.core.frame.DataFrame) and refsigs.index is not None:
        refidx = refsigs.index
    elif refidx is None:
        refidx = pd.Index([f'refsig_{i}' for i in range(0,refsigs.shape[0])])
    
    sim = cosine_similarity(sigs, refsigs)
    closest_dist = np.max(sim, axis=1)
    closest = refidx[np.argmax(sim, axis=1)]
    above_thresh = [str(refidx[x].to_numpy()) for x in sim > thresh]
    
    df=pd.DataFrame.from_dict({'inferred signature': [f'sig_{i}' for i in range(0,sigs.shape[0])],
                               'closest reference signature':closest, 
                               'dist to closest': closest_dist,
                               f'reference signatures with >{thresh} similarity': above_thresh
                              })
    df.index = [f'sig_{i}' for i in range(0,sigs.shape[0])]
    return df
