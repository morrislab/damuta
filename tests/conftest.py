import pytest 
import pathlib as pl
import pandas as pd
from damuta.base import DataSet, SignatureSet

@pytest.fixture(scope="session")
def test_data():
    return pl.Path(__file__).resolve().parent / 'test_data'

@pytest.fixture(scope="session")
def pcawg():
    counts = pd.read_csv(test_data / 'pcawg_counts.csv',  index_col=0)
    annotation = pd.read_csv(test_data / 'pcawg_cancer_types.csv', index_col=0)
    return DataSet(counts, annotation)

@pytest.fixture(scope="session")
def cosmic():
    sigs = pd.read_csv(test_data / 'COSMIC_v3.2_SBS_GRCh37.txt', sep='\t', index_col = 0).T
    return SignatureSet(sigs)