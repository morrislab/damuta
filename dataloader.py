# parse mutational catalogue from pcawg synthetic data format
import jax.numpy as jnp
from jax import random
from jax.ops import index, index_update
from jax.nn import softmax
import numpyro
import numpyro.distributions as dist
import pandas as pd
import argparse

mut_multidex = pd.MultiIndex.from_tuples([
    ('C>A', 'ACA'), ('C>A', 'ACC'), ('C>A', 'ACG'), ('C>A', 'ACT'), ('C>A', 'CCA'), ('C>A', 'CCC'), 
    ('C>A', 'CCG'), ('C>A', 'CCT'), ('C>A', 'GCA'), ('C>A', 'GCC'), ('C>A', 'GCG'), ('C>A', 'GCT'), 
    ('C>A', 'TCA'), ('C>A', 'TCC'), ('C>A', 'TCG'), ('C>A', 'TCT'), ('C>G', 'ACA'), ('C>G', 'ACC'), 
    ('C>G', 'ACG'), ('C>G', 'ACT'), ('C>G', 'CCA'), ('C>G', 'CCC'), ('C>G', 'CCG'), ('C>G', 'CCT'), 
    ('C>G', 'GCA'), ('C>G', 'GCC'), ('C>G', 'GCG'), ('C>G', 'GCT'), ('C>G', 'TCA'), ('C>G', 'TCC'), 
    ('C>G', 'TCG'), ('C>G', 'TCT'), ('C>T', 'ACA'), ('C>T', 'ACC'), ('C>T', 'ACG'), ('C>T', 'ACT'), 
    ('C>T', 'CCA'), ('C>T', 'CCC'), ('C>T', 'CCG'), ('C>T', 'CCT'), ('C>T', 'GCA'), ('C>T', 'GCC'), 
    ('C>T', 'GCG'), ('C>T', 'GCT'), ('C>T', 'TCA'), ('C>T', 'TCC'), ('C>T', 'TCG'), ('C>T', 'TCT'), 
    ('T>A', 'ATA'), ('T>A', 'ATC'), ('T>A', 'ATG'), ('T>A', 'ATT'), ('T>A', 'CTA'), ('T>A', 'CTC'), 
    ('T>A', 'CTG'), ('T>A', 'CTT'), ('T>A', 'GTA'), ('T>A', 'GTC'), ('T>A', 'GTG'), ('T>A', 'GTT'), 
    ('T>A', 'TTA'), ('T>A', 'TTC'), ('T>A', 'TTG'), ('T>A', 'TTT'), ('T>C', 'ATA'), ('T>C', 'ATC'), 
    ('T>C', 'ATG'), ('T>C', 'ATT'), ('T>C', 'CTA'), ('T>C', 'CTC'), ('T>C', 'CTG'), ('T>C', 'CTT'), 
    ('T>C', 'GTA'), ('T>C', 'GTC'), ('T>C', 'GTG'), ('T>C', 'GTT'), ('T>C', 'TTA'), ('T>C', 'TTC'), 
    ('T>C', 'TTG'), ('T>C', 'TTT'), ('T>G', 'ATA'), ('T>G', 'ATC'), ('T>G', 'ATG'), ('T>G', 'ATT'), 
    ('T>G', 'CTA'), ('T>G', 'CTC'), ('T>G', 'CTG'), ('T>G', 'CTT'), ('T>G', 'GTA'), ('T>G', 'GTC'), 
    ('T>G', 'GTG'), ('T>G', 'GTT'), ('T>G', 'TTA'), ('T>G', 'TTC'), ('T>G', 'TTG'), ('T>G', 'TTT')])

# https://stackoverflow.com/a/37755413
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = jnp.array(values)
        return super().__call__(parser, namespace, values, option_string)

def catalogue_to_Y(catalogue):
    # catalogue should be in alphabetical order
    assert catalogue.shape[0] == 96
    assert all(catalogue.index == mut_multidex)

    # split by mutation type, transpose and 0-pad
    cat = [x.T for x in jnp.array(catalogue).split(6)]
    padding = jnp.zeros((catalogue.shape[1],16), dtype=int)

    cat = [jnp.concatenate([cat[0], padding], axis = 1), 
           jnp.concatenate([cat[1], padding], axis = 1),
           jnp.concatenate([cat[2], padding], axis = 1),
           jnp.concatenate([padding, cat[3]], axis = 1),
           jnp.concatenate([padding, cat[4]], axis = 1),
           jnp.concatenate([padding, cat[5]], axis = 1)]

    Y = jnp.moveaxis(jnp.stack(cat),0,2)
    assert Y.shape == ((catalogue.shape[1]), 32, 6)
    
    return Y


def Y_to_catalogue(df):
    raise NotImplementedError

#    data = catalogue_to_Y(pd.read_csv(args.catalogue_fn, index_col = [0,1]).iloc[:,0:10])

    # update args based on passed data
#    vars(args)['S'] = data.shape[0]
#    vars(args)['N'] = data.sum([1,2])


def mask_renorm(B):
    # remove mutations impossible in the context and normalize
    S, C, M = B.shape
    c_mask = jnp.tile(jnp.array([False, False, False, True, True, True]), (S, C//2, 1))
    t_mask = jnp.tile(jnp.array([True, True, True, False, False, False]), (S, C//2, 1))
    sel = jnp.concatenate([c_mask, t_mask], axis = 1)
    B = index_update(B, index[sel], 0.)
    return (B/B.sum(-1)[:,:, jnp.newaxis])

def generate_data(args, key):
    key, *subkeys = random.split(key, 15)

    theta = dist.Dirichlet(args.psi).sample(subkeys[0], (args.S,))
    A = dist.Dirichlet(args.gamma).sample(subkeys[1], (args.S, args.J))
    phi = dist.Dirichlet(args.alpha).sample(subkeys[2], (args.J,))
    etaC = dist.Dirichlet(args.betaC).sample(subkeys[3], (args.K,))
    etaT = dist.Dirichlet(args.betaT).sample(subkeys[3], (args.K,))

    # for each sample get count of damage contexts drawn from each damage context signature
    X = dist.MultinomialProbs(theta @ phi, args.N).sample(subkeys[7], (1,)).squeeze()

    # get transition probabilities
    B_C = softmax(phi.T[0:16] @ A @ etaC)
    B_T = softmax(phi.T[16:32] @ A @ etaT)

    # for each damage context, get count of misrepair
    Y_C = dist.MultinomialProbs(B_C, X[:,0:16]).sample(subkeys[8], (1,)).reshape(args.S, -1)
    Y_T = dist.MultinomialProbs(B_C, X[:,16:32]).sample(subkeys[9], (1,)).reshape(args.S, -1)
    
    Y = jnp.concatenate([Y_C, Y_T], axis=1)
    
    return Y

def get_betas(args):
    # calculate betaC and betaT from passed arguments. beta takes priority over bprime.
    # single values will be expanded to all dimensions, ie. uniform concentration vector
    assert args.M == 6, "Parameter M only makes sense for M=6. recieved M={args.M}"

    if(args.beta is not None):
        assert args.beta.shape == (1,) or args.beta.shape == (args.M,),\
        f"Paramteter beta misized; betaC and betaT misspecified. beta.shape should be either (1,) or (M,). recieved beta.shape={args.beta.shape}, M={args.M}"

        if args.beta.shape == (1,): return jnp.full(args.M//2, args.beta[0]), jnp.full(args.M//2, args.beta[0])
        else: return args.beta

    else:
        assert args.bprime.shape == (1,) or args.bprime.shape == (4,), \
        f"Parameter bprime misized; bprime.shape should be either () or (4,). recieved bprime.shape={args.bprime.shape}"

        # expand dimensions if applicable
        if args.bprime.shape == (1,):
                vars(args)['bprime'] = jnp.full(4, args.bprime[0])
    
        # set beta from beta'
        return jnp.array([args.bprime[0],args.bprime[2],args.bprime[3]]), \
               jnp.array([args.bprime[0],args.bprime[1],args.bprime[2],])

def validate_args(args, data=None):
    # validate passed arguments
    #args = parser.parse_args()

    # expand dimensions where applicable
    for va, vl, ka, kl in zip([args.N, args.psi, args.gamma, args.alpha], \
                              [args.S, args.J,   args.K,     args.C],    \
                              ["N", "psi", "gamma", "alpha"], ["S", "J", "K", "C"]):

        assert va.shape == (1,) or va.shape[0] == vl, \
        f"Parameter {ka} misized; {ka}.shape should be either (1,) or ({kl},). recieved {ka}.shape={va.shape}, {kl}={vl}"

        if va.shape == (1,):
            vars(args)[f'{ka}'] = jnp.full(vl, va[0])

    # compute betaC and betaT  
    vars(args)['betaC'], vars(args)['betaT'] = get_betas(args)

    # correct any args mismatched with data

    return args