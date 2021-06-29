from config import *
import numpy as np
import pymc3 as pm
import pandas as pd
import theano.tensor as tt 
import typing
import wandb
from scipy.special import softmax, logsumexp, loggamma

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


def load_sigs(fn, naming_style = 'pcawg', **kwargs):
    styles = ['pcawg', 'cosmic']
    assert naming_style in styles, f'naming_style should be one of {styles}'
    
    
    if naming_style == 'cosmic':
        # read in sigs
        sigs = pd.read_csv(fn, index_col = 0, **kwargs).reindex(mut96)
        # sanity check for matching mut96, should have no NA 
        sel = (~sigs.isnull()).all(axis = 1)
        assert sel.all(),\
            f'invalid signature definitions: null entry for types {list(sigs.index[~sel])}' 
        # convert to pcawg convention
        sigs = sigs.set_index(idx96)
        
    else:
        # read in sigs
        sigs = pd.read_csv(fn, index_col = (0,1), **kwargs).reindex(idx96)
        # sanity check for idx, should have no NA
        sel = (~sigs.isnull()).all(axis = 1)
        assert sel.all(),\
            f'invalid signature definitions: null entry for types {list(sigs.index[~sel])}' 
        
    # check colsums are 1
    sel = np.isclose(sigs.sum(axis=0), 1)
    assert sel.all() ,\
        f'invalid signature definitions: does not sum to 1 in columns {list(sigs.columns[~sel])}' 
        
    return sigs

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


def split_count(counts, fraction):
    c = (counts * fraction).astype(int)
    frac1 = np.histogram(np.repeat(np.arange(96), c), bins=96, range=(0, 96))[0]
    frac2 = counts - frac1
    assert all(frac2 >= 0) and all(frac1 >= 0)
    return frac1, frac2

def split_by_count(data, fraction=0.8):
    stacked = np.array([split_count(m, fraction) for m in data])
    return stacked[:,0,:], stacked[:,1,:]

def split_by_S(data, fraction=0.8):
    c = int((data.shape[0] * fraction))
    frac1 = data[0:c]
    frac2 = data[c:(data.shape[0])]
    return frac1, frac2

def get_tau(phi, eta):
    assert len(phi.shape) == 2 and len(eta.shape) == 3
    tau =  np.einsum('jpc,pkm->jkpmc', phi.reshape((-1,2,16)), eta).reshape((-1,96))
    return tau

def get_phis(sigs):
    # for each signature in sigs, get the corresponding phi
    wrapped = sigs.reshape(sigs.shape[0], -1, 16)
    phis = np.hstack([wrapped[:,[0,1,2],:].sum(1), wrapped[:,[3,4,5],:].sum(1)])
    return phis

def get_etas(sigs):
    # for each signature in sigs, get the corresponding eta
    # ordered alphabetically, or pyrimidine last (?)
    wrapped = sigs.reshape(sigs.shape[0], -1, 16)
    etas = wrapped.sum(2)
    return etas#[0,1,2,3,5,4]

def flatten_eta(eta): # eta pkm -> kc
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
    