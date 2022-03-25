import pandas as pd
from damuta.utils import load_sigs

# sanity check that package and test_data/ are correctly loaded.

def test_paths(test_data):
    sig_defs = pd.read_csv(test_data / 'COSMIC_v3.2_SBS_GRCh37.txt', sep='\t', index_col = 0)
    assert sig_defs.T.shape == load_sigs(test_data / 'COSMIC_v3.2_SBS_GRCh37.txt').shape
    