import pytest
import numpy as np
from damut import load_sigs, sim_from_sigs, sim_parametric

sig_defs = load_sigs('data/COSMIC_v3.2_SBS_GRCh37.txt')

def test_from_sigs_seeding():
    # same seed 
    d0, p0 = sim_from_sigs(sig_defs, 0.1, 10, 1000, 5, np.random.default_rng(100))
    d1, p1 = sim_from_sigs(sig_defs, 0.1, 10, 1000, 5, np.random.default_rng(100))
    assert np.all(d0==d1) and np.all([np.all(p0[x] == p1[x]) for x in p0.keys()]), 'Seeding not reproducible'
    # diff seed 
    d0, p0 = sim_from_sigs(sig_defs, 0.1, 10, 1000, 5)
    d1, p1 = sim_from_sigs(sig_defs, 0.1, 10, 1000, 5)
    assert not (np.all(d0==d1) and np.all([np.all(p0[x] == p1[x]) for x in p0.keys()])), 'Different seeds produced same result'

def test_parametric_seeding():
    d0, p0 = sim_parametric(5,6,10,1000,.1,.1,.1,.1,np.random.default_rng(100))
    d1, p1 = sim_parametric(5,6,10,1000,.1,.1,.1,.1,np.random.default_rng(100))
    assert np.all(d0==d1) and np.all([np.all(p0[x] == p1[x]) for x in p0.keys()]), 'Seeding not reproducible'
    d0, p0 = sim_parametric(5,6,10,1000,.1,.1,.1,.1)
    d1, p1 = sim_parametric(5,6,10,1000,.1,.1,.1,.1)
    assert not ( np.all(d0==d1) and np.all([np.all(p0[x] == p1[x]) for x in p0.keys()]) ), 'Different seeds produced same result'
    