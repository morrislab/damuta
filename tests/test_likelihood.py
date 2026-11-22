"""Tests for the math in models.mutation_probabilities.
"""
import numpy as np
import pytensor.tensor as pt
import pytest

from damuta.models import TandemLda, mutation_probabilities
from damuta.utils import get_tau

S, J, K = 5, 4, 3


@pytest.fixture
def params():
    """Random theta (S,J), A (S,J,K), phi (J,32), eta (K,2,3)."""
    rng = np.random.default_rng(0)
    theta = rng.dirichlet(np.ones(J), size=S)
    A = rng.dirichlet(np.ones(K), size=(S, J))
    phi = rng.dirichlet(np.ones(32), size=J)
    eta = rng.dirichlet(np.ones(3), size=(K, 2))
    return {"theta": theta, "A": A, "phi": phi, "eta": eta}


def calc_mut_probs(theta, A, phi, eta, likelihood):
    """Evaluate mutation_probabilities, shape (S, 96)."""
    return pt.as_tensor_variable(
        mutation_probabilities(theta, A, phi, eta, likelihood)
    ).eval()


@pytest.mark.parametrize("likelihood", ["sequential", "product"])
def test_rows_sum_to_one(params, likelihood):
    """Each sample's 96 mutation probabilities sum to 1."""

    mut_probs = calc_mut_probs(
        params["theta"], params["A"], params["phi"], params["eta"], likelihood
    )
    np.testing.assert_allclose(mut_probs.sum(axis=1), 1.0)


def test_sequential_matches_numpy(params):
    """Sequential B equals W @ tau computed directly in numpy."""

    W = params["theta"][:, :, None] * params["A"]
    tau = get_tau(params["phi"], params["eta"])
    B = np.dot(W.reshape((S, J * K)), tau)

    mut_probs = calc_mut_probs(
        params["theta"], params["A"], params["phi"], params["eta"], "sequential"
    )

    np.testing.assert_allclose(mut_probs, B)


def test_product_matches_numpy(params):
    """Product B equals the published formula written in numpy."""

    context = np.dot(params["theta"], params["phi"])
    misrepair = np.einsum("sj,sjk->sk", params["theta"], params["A"])
    substitution = np.einsum("sk,kpm->spm", misrepair, params["eta"])
    B = np.einsum(
        "spm,spc->spmc", substitution, context.reshape((S, 2, 16))
    ).reshape((S, 96))

    mut_probs = calc_mut_probs(
        params["theta"], params["A"], params["phi"], params["eta"], "product"
    )

    np.testing.assert_allclose(mut_probs, B)


def test_only_sequential_form_distinguishes_A(params):
    """Identifiability: Sequential form distinguishes between two different A
    with the same theta-weighted column sums.
    """

    A2 = params["A"].copy()
    for s in range(S):
        size = 0.5 * np.min(params["theta"][s, :2, None] * A2[s, :2, :2])
        delta = size / params["theta"][s, 0:2]
        A2[s, 0:2, 0] += delta * np.array([1, -1])
        A2[s, 0:2, 1] += delta * np.array([-1, 1])

    np.testing.assert_allclose(A2.sum(axis=2), 1.0)
    assert (A2 > 0).all()

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(params["A"], A2)

    misrepair_A = np.einsum("sj,sjk->sk", params["theta"], params["A"])
    misrepair_A2 = np.einsum("sj,sjk->sk", params["theta"], A2)

    np.testing.assert_allclose(misrepair_A, misrepair_A2)

    prod_mut_probs_A = calc_mut_probs(
        params["theta"], params["A"], params["phi"], params["eta"], "product"
    )
    prod_mut_probs_A2 = calc_mut_probs(
        params["theta"], A2, params["phi"], params["eta"], "product"
    )

    seq_mut_probs_A = calc_mut_probs(
        params["theta"], params["A"], params["phi"], params["eta"], "sequential"
    )
    seq_mut_probs_A2 = calc_mut_probs(
        params["theta"], A2, params["phi"], params["eta"], "sequential"
    )

    np.testing.assert_allclose(prod_mut_probs_A, prod_mut_probs_A2)

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(seq_mut_probs_A, seq_mut_probs_A2)


def test_unknown_likelihood_raises(params):
    """A misspelled likelihood name raises ValueError."""

    with pytest.raises(ValueError):
        calc_mut_probs(
            params["theta"], params["A"], params["phi"], params["eta"],
            "product__"
        )


def test_model_rejects_unknown_likelihood(pcawg):
    """TandemLda checks the likelihood name when it's created, before any fit."""

    with pytest.raises(ValueError):
        model = TandemLda(
            dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5,
            init_strategy='uniform', likelihood='product__'
        )
