# gen.py 
# generative process for substitution mutations in cancer
import argparse
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

import numpy as np
from numpy.random import default_rng
rng = default_rng(1080)


# https://stackoverflow.com/a/37755413
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = jnp.array(values)
        return super().__call__(parser, namespace, values, option_string)

def mask_renorm(B):
    # remove mutations impossible in the context and normalize
    S, C, M = B.shape
    c_mask = np.tile(np.array([False, False, False, True, True, True]), (S, C//2 ,1))
    t_mask = np.tile(np.array([True, True, True, False, False, False]), (S, C//2 ,1))
    m = np.concatenate([c_mask, t_mask], axis = 1)
    B[m] = 0
    return B/B.sum(2)[:, :, np.newaxis]

def generate_data(args, key):
    key, *subkeys = random.split(key, 9)

    theta = dist.Dirichlet(args.psi).sample(subkeys[0], (args.S,))
    A = dist.Dirichlet(args.gamma).sample(subkeys[1], (args.S, args.J))
    phi = dist.Dirichlet(args.alpha).sample(subkeys[2], (args.J,))
    etaC = dist.Dirichlet(args.betaC).sample(subkeys[3], (args.K,))
    etaT = dist.Dirichlet(args.betaT).sample(subkeys[4], (args.K,))
    eta = jnp.concatenate([etaC, etaT], axis = 1)

    # for each sample get count of damage contexts drawn from each damage context signature
    #X = np.array([rng.multinomial(*x) for x in zip(args.N, np.dot(theta, phi), np.full((args.S,), 1))]).squeeze()
    X_p = dist.MultinomialProbs(jnp.dot(theta, phi), args.N)
    X = X_p.sample(subkeys[5], (1,)).squeeze()  
    # get transition probabilities
    B = mask_renorm(np.dot(phi.T, np.dot(A, eta)).swapaxes(0,1))
    # for each damage context, get count of misrepair
    Y = dist.MultinomialProbs(B, X).sample(subkeys[6], (1,)).squeeze()


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


def validate_args(parser):
    # validate passed arguments
    args = parser.parse_args()

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

    return args


def main():

    parser = argparse.ArgumentParser(description='damut generative model')
    parser.add_argument("-S", type=int, default = 100, help = "number of samples")
    parser.add_argument('-N', type=int, default = jnp.array([1000]), nargs='+', 
                        action=Store_as_array, help = "number of mutations per sample")
    parser.add_argument('-C', type=int, default = 32, help = "number of damage context types")
    parser.add_argument('-M', type=int, default = 6, help = "number of misrepair types")
    parser.add_argument('-J', type=int, default = 10, help = "number of damage context signatures")
    parser.add_argument('-K', type=int, default = 10, help = "number of misrepair signatures")
    parser.add_argument('-psi', type=float, default = jnp.array([1.0]), nargs='+', action=Store_as_array,
                        help='psi parameter for selecting sample-wise damage context signature activities')
    parser.add_argument('-gamma', type=float, default = jnp.array([0.1]), nargs='+', action=Store_as_array,
                        help='gamma parameter for selecting sample-wise misrepair signature activities')
    parser.add_argument('-alpha', type=float, default = jnp.array([0.1]), nargs='+', action=Store_as_array,
                        help='alpha parameter for damage context signature')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-beta', type=float, nargs='+', action=Store_as_array,
                        help='beta parameter for misrepair signature; explicitly set betaC and betaT')
    group.add_argument('-bprime', type=float, default = jnp.array([0.1]), nargs='+', action=Store_as_array,
                        help='beta prime hyperparameter that defines betaC and betaT via each base')
    parser.add_argument('-seed', type=int, default = 0, help='rng seed')

    subparsers = parser.add_subparsers(help='help for subcommand')
    
    args = validate_args(parser)
    key = random.PRNGKey(args.seed)
    data = generate_data(args, key)

    print(data[0].sum(), data[1].sum(), data[2].sum())
if __name__ == '__main__':
    main()