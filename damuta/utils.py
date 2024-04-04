import numpy as np
import pymc3 as pm
import pandas as pd
#import pickle
#import wandb
#import yaml
import warnings
from sklearn.cluster import k_means
from scipy.special import softmax, logsumexp, loggamma
from sklearn.metrics.pairwise import cosine_similarity
from .constants import * 
import pkg_resources
from scipy.stats import multinomial
from scipy.optimize import linear_sum_assignment

# constants
#C=32
#M=3
#P=2

def dirichlet(node_name, a, shape, scale=1, testval=None, observed=None):
    # dirichlet reparameterized here because of stickbreaking bug
    # https://github.com/pymc-devs/pymc3/issues/4733
    X = pm.Gamma(f'gamma_{node_name}', mu = a, sigma = scale, shape = shape, testval = testval, observed = observed)
    Y = pm.Deterministic(node_name, (X/X.sum(axis = (X.ndim-1))[...,None]))
    return Y


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
    
    ## handle seeding
    #config['dataset'].update({'data_rng': np.random.default_rng(config['dataset']['data_seed'])})
    #config['model'].update({'model_rng': np.random.default_rng(config['model']['model_seed'])})
    
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
    warnings.warn("load_sigs is deprecated, see Damuta class", DeprecationWarning)

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
    warnings.warn("load_counts is deprecated, see Damuta class", DeprecationWarning)
    counts = pd.read_csv(counts_fp, index_col = 0, header = 0)[mut96]
    assert counts.ndim == 2, 'Mutation counts failed to load. Check column names are mutation type (ex. A[C>A]A). See COSMIC database for more.'
    assert counts.shape[1] == 96, f'Expected 96 mutation types, got {counts.shape[1]}'
    return counts

def subset_samples(dataset, annotation, annotation_subset, sel_idx = 0):
    # subset sample ids by matching to annotation_subset
    # expect annotation_subset to be pd dataframe with ids as index

    if annotation_subset is None:
        return dataset, annotation

    # stop string being auto cast to list
    if type(annotation_subset) == str:
        annotation_subset = [annotation_subset]
    
    if annotation.ndim > 2:
        warnings.warn(f"More than one annotation is available per sample, selection index {sel_idx}", UserWarning)
        
    # annotation ids should match sample ids
    assert dataset.index.isin(annotation.index).any(), 'No sample ID matches found in dataset for the provided annotation'

    # reoder annotation (with gaps) to match dataset
    annotation = annotation.reindex(dataset.index)

    # partial matches allowed
    sel = np.fromiter((map(any, zip(*[annotation[annotation.columns[sel_idx]].str.contains(x) for x in annotation_subset] ))), dtype = bool)
        
    # type should appear in the type column of the lookup 
    assert sel.any(), 'Cancer type subsetting yielded no selection. Check keywords?'

    dataset = dataset.loc[annotation.index[sel]]
    annotation = annotation.loc[annotation.index[sel]]
    return dataset, annotation

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

def get_eta(sigs, normalize=True):
    # for each signature in sigs, get the corresponding eta 
    wrapped = sigs.reshape(-1, 6, 16)
    eta = wrapped.sum(2).reshape(-1,3)
    if normalize:
        # normalize such that etaC and etaT sum to 1 respectively.
        eta = (eta/eta.sum(1)[:,None])
    return eta.reshape(-1,6)

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

def mult_ll(x, p):
    # x and p should both be same dimensions; Sx96
    return loggamma(x.sum(1) + 1) - loggamma(x+1).sum(1) + (x * np.log(p)).sum(1)

def alp_B(data, B):
    return mult_ll(data, B).sum()

def lap_B(data, Bs):
    # Bs should be shape DxSx96 where D is the number of posterior samples
    # use logsumexp for stability
    assert Bs.ndim == 3, 'expected multiple trials for B'
    return logsumexp(np.vstack([mult_ll(data, B) for B in Bs]).sum(1)) - np.log(Bs.shape[0])

def profile_sigs(est_sigs, ref_sigs, thresh = 0.95):
    # return mapping of sigs to similar refsigs 
    # get most similar sig
    dists = 1- cosine_similarity(est_sigs.to_numpy(), ref_sigs.to_numpy())
    profile = pd.DataFrame({
        'closest' : ref_sigs.index[dists.argmin(1)],
        'closest_sim' : 1-dists.min(1),
        }, index = est_sigs.index)
    # hungarian algorithm to assign sigs uniquely
    row_ind, col_ind = linear_sum_assignment(dists)
    profile['hungarian_matched']  = profile.index.map({est_sigs.index[i]: (ref_sigs.index[col_ind]).to_list() for i in row_ind})
    profile['hungarian_matched_sim'] = profile.index.map({est_sigs.index[i]: (1-(dists[row_ind, col_ind]))[row_ind==i][0] for i in row_ind})
    # get any other sigs >=0.95 sim
    est_ind, ref_ind = np.where(dists<(1-0.05))
    profile['geq_95_sim'] = profile.index.map({est_sigs.index[i]: (ref_sigs.index[ref_ind])[est_ind == i].tolist() for i in np.unique(est_ind)}.get)
    return profile


def load_cosmic_V3():
    """Return a dataframe of COSMIC V3 signature definitions

    Contains:
        96 mutation type columns of non-null float64
        78 rows of signature definitions, rows sum to 1

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    f = pkg_resources.resource_filename(__name__, 'data/COSMIC_v3.2_SBS_GRCh37.txt')
    return load_sigs(f)

def load_default_config():
    """Return a default configuration dict

    Contains:
        96 mutation type columns of non-null float64
        78 rows of signature definitions, rows sum to 1

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    f = pkg_resources.resource_filename(__name__, 'config/default.yaml')
    return load_config(f)

