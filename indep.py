# gen.py 
# generative process for substitution mutations in cancer
import argparse
from jax import random, jit
import jax.numpy as jnp
from jax.nn import softmax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
from dataloader import *
import yaml
from scipy.stats import wasserstein_distance 
numpyro.set_platform('gpu')

import wandb
wandb.init(project="damut")
import os
os.environ['WANDB_MODE'] = 'dryrun'

def model(data, args):
    # posterior approximation q(z|x)
    alpha_q = numpyro.param("context_type_bias", jnp.zeros((args.C,))+1, constraint=dist.constraints.positive)
    psi_q = numpyro.param("context_sig_bias", jnp.zeros((args.J,))+0.5, constraint=dist.constraints.positive)
    gamma_q = numpyro.param("misrepair_sig_bias", jnp.zeros((args.K,))+0.5, constraint=dist.constraints.positive)
    betaC_q = numpyro.param("misrepair_C_bias", jnp.zeros((args.M//2,))+0.5, constraint=dist.constraints.positive)
    betaT_q = numpyro.param("misrepair_T_bias", jnp.zeros((args.M//2,))+0.5, constraint=dist.constraints.positive)

    with numpyro.plate("J", args.J):
        phi = numpyro.sample("context_defs", dist.Dirichlet(alpha_q))

    with numpyro.plate("K", args.K):
        etaC = numpyro.sample("C_defs", dist.Dirichlet(betaC_q))
        etaT = numpyro.sample("T_defs", dist.Dirichlet(betaT_q))
        
    with numpyro.plate("thetaS", args.S):
        theta = numpyro.sample("context_activities", dist.Dirichlet(psi_q))
    
    with numpyro.plate("J", args.J):
        with numpyro.plate("S", args.S):
            A = numpyro.sample("misrepair_activites", dist.Dirichlet(gamma_q))
        
    
    # https://stackoverflow.com/a/47625092
    # https://stackoverflow.com/a/36031086
    BC = ((theta@phi)[:, None]) * ((jnp.einsum('ij,ijk->ik', theta, A)@etaC)[..., None])
    BC = jnp.moveaxis(BC,1,2).reshape((args.S, -1))
    BT = ((theta@phi)[:, None]) * ((jnp.einsum('ij,ijk->ik', theta, A)@etaT)[..., None])
    BT = jnp.moveaxis(BT,1,2).reshape((args.S, -1))
    
    B = jnp.concatenate([BC[:,0:48], BT[:,48:96]], axis = 1)
    
    # for each damage context, get count of misrepair
    Y = numpyro.sample("mutation", dist.MultinomialProbs(B, args.N), obs = data)

def log_metrics(args, svi, state, loss):
    wandb.config.update(args)
    wandb.log({"loss": float(loss)})
    params = svi.get_params(state)
    wandb.log({"alpha earthmover error": wasserstein_distance(args.psi, params['context_type_bias'])})
    wandb.log({"psi earthmover error": wasserstein_distance(args.psi, params['context_sig_bias'])})
    wandb.log({"gamma earthmover error": wasserstein_distance(args.psi, params['misrepair_sig_bias'])})
    wandb.log({"beta C earthmover error": wasserstein_distance(args.psi, params['misrepair_C_bias'])})
    wandb.log({"beta T earthmover error": wasserstein_distance(args.psi, params['misrepair_T_bias'])})
    #wandb.log({"max theta": max(params['context_activities'])})
    
def run_inference(data, args, key):
    key, *subkeys = random.split(key, 2)
    numpyro.enable_validation()
    optimizer = numpyro.optim.Adam(step_size=0.01)
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    #svi_result = svi.run(subkeys[0], args.nsteps, data, args)
    
    def body_fn(svi_state, carry=None):
        svi_state, loss = svi.update(svi_state, data, args)
        return svi_state, loss


    svi_state = svi.init(subkeys[0], data, args)

    for i in range(args.nsteps):
        svi_state, loss = jit(body_fn)(svi_state)
        
        log_metrics(args, svi, svi_state, loss)

def main():

    parser = argparse.ArgumentParser(description='damut generative model')
    parser.add_argument("-S", type=int, default = 100, help = "number of samples")
    parser.add_argument('-N', type=int, default = jnp.array([1000]), nargs='+', 
                        action=Store_as_array, help = "number of mutations per sample (1xS)")
    parser.add_argument('-C', type=int, default = 32, help = "number of damage context types")
    parser.add_argument('-M', type=int, default = 6, help = "number of misrepair types")
    parser.add_argument('-J', type=int, default = 10, help = "number of damage context signatures")
    parser.add_argument('-K', type=int, default = 10, help = "number of misrepair signatures")
    parser.add_argument('-psi', type=float, default = jnp.array([1.0]), nargs='+', action=Store_as_array,
                        help='psi parameter for selecting sample-wise damage context signature activities (1xJ)')
    parser.add_argument('-gamma', type=float, default = jnp.array([0.1]), nargs='+', action=Store_as_array,
                        help='gamma parameter for selecting sample-wise misrepair signature activities (1xK)')
    parser.add_argument('-alpha', type=float, default = jnp.array([0.1]), nargs='+', action=Store_as_array,
                        help='alpha parameter for damage context signature (1xC)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-beta', type=float, nargs='+', action=Store_as_array,
                        help='beta parameter for misrepair signature; explicitly set betaC and betaT (1x6)')
    group.add_argument('-bprime', type=float, default = jnp.array([0.1]), nargs='+', action=Store_as_array,
                        help='beta prime hyperparameter that defines betaC and betaT via each base (1xM)')
    parser.add_argument('-seed', type=int, default = 0, help='rng seed')
    parser.add_argument('-nsteps', type=int, default = 100, help='inference iterations')
    
    
    args = parser.parse_args()
    args = validate_args(args, data = None)
    key = random.PRNGKey(args.seed)
    data = generate_data(args, key)
    #data = catalogue_to_Y(pd.read_csv(args.catalogue_fn, index_col = [0,1]).iloc[:,0:10])
    

    run_inference(data, args, key)
    

if __name__ == '__main__':
    main()
