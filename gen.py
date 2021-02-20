# gen.py 
# generative process for substitution mutations in cancer
import argparse
from jax import random, jit
import jax.numpy as jnp
from jax.ops import index, index_update
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
import wandb
wandb.init(project="damut")
import os
os.environ['WANDB_MODE'] = 'dryrun'
numpyro.set_platform('gpu')

# https://stackoverflow.com/a/37755413
class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = jnp.array(values)
        return super().__call__(parser, namespace, values, option_string)

def mask_renorm(B):
    # remove mutations impossible in the context and normalize
    S, C, M = B.shape
    c_mask = jnp.tile(jnp.array([False, False, False, True, True, True]), (S, C//2, 1))
    t_mask = jnp.tile(jnp.array([True, True, True, False, False, False]), (S, C//2, 1))
    m = jnp.concatenate([c_mask, t_mask], axis = 1)
    B = index_update(B, index[m], 0.)
    return B/B.sum(2)[:, :, jnp.newaxis]

def generate_data(args, key, perturb = 0):
    key, *subkeys = random.split(key, 15)

    theta = dist.Dirichlet(args.psi + perturb).sample(subkeys[0], (args.S,))
    A = dist.Dirichlet(args.gamma + perturb).sample(subkeys[1], (args.S, args.J))
    phi = dist.Dirichlet(args.alpha + perturb).sample(subkeys[2], (args.J,))
    etaC = dist.Dirichlet(args.betaC + perturb).sample(subkeys[3], (args.K,))
    etaT = dist.Dirichlet(args.betaT + perturb).sample(subkeys[3], (args.K,))
    #eta = jnp.concatenate([etaC, etaT], axis = 1)

    #etaC = jnp.concatenate([dist.Dirichlet(args.betaC).sample(subkeys[5], (args.K,)), jnp.full((args.K, args.M//2), 0)], axis = 1)
    #etaT = jnp.concatenate([jnp.full((args.K, args.M//2), 0), dist.Dirichlet(args.betaT).sample(subkeys[6], (args.K,))], axis = 1)
    
    # for each sample get count of damage contexts drawn from each damage context signature
    X = dist.MultinomialProbs(theta @ phi, args.N).sample(subkeys[7], (1,)).squeeze()

    # get transition probabilities
    #B = mask_renorm(phi.T @ A @ eta).swapaxes(0,1)

    # for each damage context, get count of misrepair
    #Y = dist.MultinomialProbs(B, X).sample(subkeys[8], (1,)).squeeze()


    #C_mask = jnp.arange(16)
    #T_mask = jnp.arange(16, 32)

    #phiC = phi[:,C_mask]
    #BC = (phiC.T @ A @ etaC)[:,C_mask,:]
    #BC = BC/BC.sum(2)[:, :, jnp.newaxis]
    #YC = dist.MultinomialProbs(BC, X[:,C_mask]).sample(subkeys[9], (1,)).squeeze()

    #phiT = phi[:,T_mask]
    #BT = (phiT.T @ A @ etaT)[:,T_mask,:]
    #BT = BT/BT.sum(2)[:, :, jnp.newaxis]
    #YT = dist.MultinomialProbs(BT, X[:,T_mask]).sample(subkeys[10], (1,)).squeeze()

    #YC[0].sum()
    #YT[0].sum()


    return X

def model(data, args):
    with numpyro.plate("J", args.J):
        phi = numpyro.sample("context_defs", dist.Dirichlet(args.alpha))

    with numpyro.plate("S", args.S):
        theta = numpyro.sample("context_activities", dist.Dirichlet(args.psi))
    
    X = numpyro.sample("context_type", dist.MultinomialProbs(theta @ phi, 1000), obs=data)

        
    #return X    

def guide(data, args):
    # posterior approximation q(z|x)
    # here args are used as prior
    alpha_q = numpyro.param("context_type_bias", args.alpha, constraint=constraints.positive)
    psi_q = numpyro.param("context_sig_bias", args.psi, constraint=constraints.positive)
    
    with numpyro.plate("J", args.J):
        phi_q = numpyro.sample("context_defs", dist.Dirichlet(alpha_q))

    with numpyro.plate("S", args.S) as s_idx:
        theta_q = numpyro.sample("context_activities", dist.Dirichlet(psi_q))
    
    numpyro.sample("context_type", dist.MultinomialProbs(theta_q @ phi_q, 1000), obs=data)


def run_inference(data, args, key):
    key, *subkeys = random.split(key, 2)
    optimizer = numpyro.optim.Adam(step_size=0.0005)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    #svi_result = svi.run(random.PRNGKey(0), args.nsteps, data, args)
    
    def body_fn(svi_state, carry=None):
            svi_state, loss = svi.update(svi_state, data, args)
            return svi_state, loss


    svi_state = svi.init(subkeys[0], data, args)
    for i in range(1, args.nsteps + 1):
                    svi_state, loss = jit(body_fn)(svi_state)
                    mean_alpha = svi.get_params(svi_state)['context_type_bias'].mean()
                    mean_psi = svi.get_params(svi_state)['context_sig_bias'].mean()
                    wandb.log({"losses": float(loss), 
                               "mean psi": float(mean_psi),
                               "mean alpha": float(mean_alpha)})
    

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

    
    args = validate_args(parser)
    wandb.config.update(args)
    key = random.PRNGKey(args.seed)
    data = generate_data(args, key, perturb = 0)
    #jnp.savez('sim_data.npz', data = data, args = args)
    #with jnp.load('sim_data.npz', allow_pickle = True) as f:
    #    data = f['data']
    run_inference(data, args, key)
    

if __name__ == '__main__':
    main()