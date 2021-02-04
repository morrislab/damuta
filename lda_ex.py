from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
numpyro.set_platform('gpu')

def model(data):
    f = numpyro.sample("latent_fairness", dist.Beta(10, 10))
    with numpyro.plate("N", data.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(f), obs=data)

def guide(data):
    alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)
    beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
                           constraint=constraints.positive)
    numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
optimizer = numpyro.optim.Adam(step_size=0.0005)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(random.PRNGKey(0), 20000, data)
params = svi_result.params
inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])