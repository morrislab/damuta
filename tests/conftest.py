import pytest 
import pathlib as pl
import pandas as pd
from damuta.base import DataSet, SignatureSet

@pytest.fixture(scope="session")
def test_data():
    return pl.Path(__file__).resolve().parent / 'test_data'

@pytest.fixture(scope="session")
def pcawg(test_data):
    counts = pd.read_csv(test_data / 'pcawg_counts.csv',  index_col=0)[0:100]
    annotation = pd.read_csv(test_data / 'pcawg_cancer_types.csv', index_col=0)[0:100]
    return DataSet(counts, annotation)

@pytest.fixture(scope="session")
def cosmic(test_data):
    sigs = pd.read_csv(test_data / 'COSMIC_v3.2_SBS_GRCh37.txt', sep='\t', index_col = 0).T
    return SignatureSet(sigs)

# Skip slow tests https://stackoverflow.com/a/61193490

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

