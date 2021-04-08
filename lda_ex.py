# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This example implements amortized Latent Dirichlet Allocation [1],
demonstrating how to marginalize out discrete assignment variables in a numpyro
model. This model and inference algorithm treat documents as vectors of
categorical variables (vectors of word ids), and collapses word-topic
assignments using numpyro's enumeration. We use PyTorch's reparametrized Gamma and
Dirichlet distributions [2], avoiding the need for Laplace approximations as in
[1]. Following [1] we use the Adam optimizer and clip gradients.

**References:**

[1] Akash Srivastava, Charles Sutton. ICLR 2017.
    "Autoencoding Variational Inference for Topic Models"
    https://arxiv.org/pdf/1703.01488.pdf
[2] Martin Jankowiak, Fritz Obermeyer. ICML 2018.
    "Pathwise gradients beyond the reparametrization trick"
    https://arxiv.org/pdf/1806.01851.pdf
"""
import argparse
import functools
import logging

import argparse
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
numpyro.set_platform('gpu')

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)




# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model(data=None, args=None, batch_size=None):
    # Globals.
    with numpyro.plate("topics", args.num_topics):
        topic_weights = numpyro.sample("topic_weights", dist.Gamma(1. / args.num_topics))
        topic_words = numpyro.sample("topic_words", dist.Dirichlet(torch.ones(args.num_words) / args.num_words))

    # Locals.
    with numpyro.plate("documents", args.num_docs) as ind:

        if data is not None:
            assert data.shape == (args.num_words_per_doc, args.num_docs), 'hello'
            data = data[:, ind]
        
        doc_topics = numpyro.sample("doc_topics", dist.Dirichlet(topic_weights))
        with numpyro.plate("words", args.num_words_per_doc):
            # The word_topics variable is marginalized out during inference
            word_topics = numpyro.sample("word_topics", dist.Categorical(doc_topics))#,infer={"enumerate": "parallel"})
            data = numpyro.sample("doc_words", dist.Categorical(topic_words[word_topics]), obs=data)

    return topic_weights, topic_words, data


# We will use amortized inference of the local topic variables, achieved by a
# multi-layer perceptron. We'll wrap the guide in an nn.Module.
def make_predictor(args):
    layer_sizes = ([args.num_words] +
                   [int(s) for s in args.layer_sizes.split('-')] +
                   [args.num_topics])
    logging.info('Creating MLP with sizes {}'.format(layer_sizes))
    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layer = nn.Linear(in_size, out_size)
        layer.weight.data.normal_(0, 0.001)
        layer.bias.data.normal_(0, 0.001)
        layers.append(layer)
        layers.append(nn.Sigmoid())
    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)


def parametrized_guide(data, args, batch_size=None):
    # Use a conjugate guide for global variables.
    topic_weights_posterior = numpyro.param(
            "topic_weights_posterior",
            lambda: torch.ones(args.num_topics),
            constraint=constraints.positive)
    topic_words_posterior = numpyro.param(
            "topic_words_posterior",
            lambda: torch.ones(args.num_topics, args.num_words),
            constraint=constraints.greater_than(0.5))
    with numpyro.plate("topics", args.num_topics):
        numpyro.sample("topic_weights", dist.Gamma(topic_weights_posterior, 1.))
        numpyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

    # Use an amortized guide for local variables.
    for i in numpyro.plate("documents", args.num_docs, batch_size):

        doc_topics = jnp.full(args.num_topics, 10)
        numpyro.sample("doc_topics", dist.Delta(doc_topics, event_dim=1))


def main(args):
    logging.info('Generating data')
    key = random.PRNGKey(0)
    key, *subkeys = random.split(key, 10)

    # sample data
    true_topic_weights = dist.Gamma(1. / args.num_topics).sample(subkeys[0], (args.num_topics,))
    true_topic_words = dist.Dirichlet(torch.ones(args.num_words) / args.num_words).sample(subkeys[1], (args.num_topics,))
    data = model(args=args)

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(args.num_docs))
    #predictor = make_predictor(args)
    #guide = functools.partial(parametrized_guide, predictor)
    Elbo = Trace_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = ClippedAdam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo, data=None, args=None, batch_size=None)

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(data, args=args, batch_size=args.batch_size)
        if step % 10 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    loss = elbo.loss(model, guide, data, args=args)
    logging.info('final loss = {}'.format(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=8, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=64, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)