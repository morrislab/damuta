import pytest 
import pymc3 as pm
import pandas as pd
import numpy as np
from damuta import load_sigs

@pytest.fixture(scope="session")
def c():
    c = pd.read_csv('data/mutation_types_raw_counts.csv', index_col=0, header=0)
    return c

@pytest.fixture(scope="session")
def a():
    a =  pd.read_csv('data/pcawg_cancer_types.csv', index_col=0, header=0)
    return a

@pytest.fixture(scope="session")
def sig_defs():
    sig_defs = load_sigs('data/COSMIC_v3.2_SBS_GRCh37.txt')
    return sig_defs

