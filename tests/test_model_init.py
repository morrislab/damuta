import pytest
import damut as da
import pandas as pd
import numpy as np

c = pd.read_csv('data/mutation_types_raw_counts.csv', index_col=0, header=0)[0:30]
sig_defs = da.load_sigs('data/COSMIC_v3.2_SBS_GRCh37.txt')

def test_kmeans():
    da.init_sigs('kmeans', data=c, J=5, K=5)

def test_from_tau():
    da.init_sigs('supply_tau', tau = sig_defs, J=5, K=5)
    
def test_random():
    da.init_sigs('random', J=5, K=5)

def test_uniform():
    rng=np.random.default_rng(100)
    assert np.all(da.init_sigs('uniform', tau = sig_defs, J=5, K=5, rng = rng) == da.init_sigs('uniform', rng = rng)), "Seeding fails with extra args"

def test_incorrect_naming():
    with pytest.raises(AssertionError):
        da.init_sigs('zilch', data=c, J=5, K=5)
    