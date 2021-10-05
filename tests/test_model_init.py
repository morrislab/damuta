import pytest
import damuta as da
import pandas as pd
import numpy as np

def test_kmeans(c):
    da.init_sigs('kmeans', data=c[0:30], J=5, K=5)

def test_from_tau(sig_defs):
    da.init_sigs('supply_tau', tau = sig_defs, J=5, K=5)
    
def test_random():
    da.init_sigs('random', J=5, K=5)

def test_uniform(sig_defs):
    rng=np.random.default_rng(100)
    assert np.all(da.init_sigs('uniform', tau = sig_defs, J=5, K=5, rng = rng) == da.init_sigs('uniform', rng = rng)), "Seeding fails with extra args"

def test_incorrect_naming(c):
    with pytest.raises(AssertionError):
        da.init_sigs('zilch', data=c, J=5, K=5)
    