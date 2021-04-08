# gen.py 
# generative process for substitution mutations in cancer
import argparse
from jax import random, jit
import jax.numpy as jnp
from jax.ops import index, index_update
from jax.nn import softmax
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.diagnostics import print_summary
from numpyro.infer import autoguide
numpyro.set_platform('gpu')

plot = False
if plot:
    import wandb
    wandb.init(project="damut")
    import os
    os.environ['WANDB_MODE'] = 'dryrun'


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
    eta = jnp.concatenate([etaC, etaT], axis = 1)

        # for each sample get count of damage contexts drawn from each damage context signature
    X = dist.MultinomialProbs(theta @ phi, args.N).sample(subkeys[7], (1,)).squeeze()

    # get transition probabilities
    B = mask_renorm(phi.T @ A @ eta)

    # for each damage context, get count of misrepair
    Y = dist.MultinomialProbs(B, X).sample(subkeys[8], (1,)).squeeze()

    return Y

def model(data, args):
    # posterior approximation q(z|x)
    alpha_q = numpyro.param("context_type_bias", jnp.zeros((args.C,))+1, constraint=constraints.positive)
    psi_q = numpyro.param("context_sig_bias", jnp.zeros((args.J,))+0.5, constraint=constraints.positive)
    gamma_q = numpyro.param("misrepair_sig_bias", jnp.zeros((args.K,))+0.5, constraint=constraints.positive)
    betaC_q = numpyro.param("misrepair_C_bias", jnp.zeros((args.M//2,))+0.5, constraint=constraints.positive)
    betaT_q = numpyro.param("misrepair_T_bias", jnp.zeros((args.M//2,))+0.5, constraint=constraints.positive)

    with numpyro.plate("J", args.J):
        phi = numpyro.sample("context_defs", dist.Dirichlet(alpha_q))
        with numpyro.plate("S", args.S):
            A = numpyro.sample("misrepair_activites", dist.Dirichlet(gamma_q))

    with numpyro.plate("K", args.K):
        etaC = numpyro.sample("C_bias", dist.Dirichlet(betaC_q))
        etaT = numpyro.sample("T_bias", dist.Dirichlet(betaT_q))
        
    with numpyro.plate("S", args.S):
        theta = numpyro.sample("context_activities", dist.Dirichlet(psi_q))
    
    # counts of damage context across samples
    X = numpyro.sample("context_type", dist.MultinomialProbs(theta @ phi, args.N))

    # mask out invalid probabilities
    maskC = jnp.concatenate([jnp.ones((args.C//2, args.M//2), dtype=bool), jnp.zeros((args.C//2, args.M//2), dtype=bool)])
    maskT = jnp.flip(maskC)
    
    # C mutation transition probabilities
    bC = softmax(phi.T @ A @ etaC, axis = -1)
    bC = jnp.where(maskC, bC, 0)
    
    # T mutation transition probabilities
    bT = softmax(phi.T @ A @ etaT, axis = -1)
    bT = jnp.where(maskT, bT, 0)
    
    B = jnp.concatenate([bC, bT], axis = -1)
    
    # for each damage context, get count of misrepair
    Y = numpyro.sample("mutation", dist.MultinomialProbs(B, X), obs = data)

def pass_guide(data, args):
    pass

def param_dist(param, name):
    labels = [f'{name}_{i}' for i in range(param.shape[0])]
    data = [[label, val] for (label, val) in zip(labels, param)]
    table = wandb.Table(data=data, columns = ["label", "value"])
    return table

def run_inference(data, args, key):
    key, *subkeys = random.split(key, 2)
    numpyro.enable_validation()
    optimizer = numpyro.optim.Adam(step_size=0.01)
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    #svi_result = svi.run(random.PRNGKey(0), args.nsteps, data, args)
    
    def body_fn(svi_state, carry=None):
            svi_state, loss = svi.update(svi_state, data, args)
            return svi_state, loss


    svi_state = svi.init(subkeys[0], data, args)

    for i in range(1, args.nsteps + 1):
        svi_state, loss = jit(body_fn)(svi_state)

        if plot: wandb.log({"loss": float(loss)})

    #post = guide.sample_posterior(random.PRNGKey(1), svi_state, (1000,))
    #print_summary(post, 0.89, False)
    print(svi.get_params(svi_state)['context_sig_bias'], 'psi')
    if plot:
        psi_hat = param_dist(svi.get_params(svi_state)['context_sig_bias'], 'psi')
        alpha_hat = param_dist(svi.get_params(svi_state)['context_type_bias'], 'alpha')
        wandb.log({"alpha hat" : wandb.plot.bar(alpha_hat, "label", "value", title="Estimated alpha values"),
                   "psi hat" : wandb.plot.bar(psi_hat, "label", "value", title="Estimated psi values")
                  })

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
    #parser.add_argument('-perturb', type=float, default = 0, help='just mess with the data a little')

    
    args = validate_args(parser)
    if plot : wandb.config.update(args)
    key = random.PRNGKey(args.seed)
    data = generate_data(args, key)
    #jnp.savez('sim_data.npz', data = data, args = args)
    #with jnp.load('sim_data.npz', allow_pickle = True) as f:
    #    data = f['data']
    run_inference(data, args, key)
    

if __name__ == '__main__':
    main()