import argparse
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
numpyro.set_platform('gpu')

def gen_data(args):
	data = jnp.concatenate([jnp.ones(args.J), jnp.zeros(args.K)])
	return data

def model(data, args):
    f = numpyro.sample("latent_fairness", dist.Beta(args.b, args.b))
    with numpyro.plate("N", data.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

def guide(data, args):
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
    parser.add_argument("-b", type=int, default = 10, help = "beta param")
    parser.add_argument("-J", type=int, default = 6, help = "data dim 0")
    parser.add_argument("-K", type=int, default = 4, help = "data dim 1")

    args = parser.parse_args()
    data = gen_data(args)
    train(data, args)

if __name__ == '__main__':
    main()