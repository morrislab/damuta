import argparse
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
numpyro.set_platform('gpu')

def gen_data(args):
	with load(args.data_source) as source:
        data = source['data']
        data_generating_args = source['args']
	return data, data_generating_args

def model(data, args):
    ca = numpyro.sample("context_activities", dist.Beta(args.b, args.b))
    ma = numpyro.sample("misrepair_activities", dist.Beta(args.b, args.b))
    with numpyro.plate("N", data.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

def guide(data, args):
    # posterior approximation q(z|x)
    alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
    beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
                           constraint=constraints.positive)
    numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

def train(data, args):
    #data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
    optimizer = numpyro.optim.Adam(step_size=0.0005)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(random.PRNGKey(0), 20000, data, args)
    params = svi_result.params
    inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])



def main():

    parser = argparse.ArgumentParser(description='damut generative model')
    parser.add_argument("data_source", type=str, 
        help = "source to read data from (npz file containing both the data and data generating parameters)")
    parser.add_argument("-ca", type=int, default = 10, help = "beta param")
    parser.add_argument("-ma", type=int, default = 6, help = "data dim 0")
    

    args = parser.parse_args()
    data = gen_data(args)
    train(data, args)

if __name__ == '__main__':
    main()