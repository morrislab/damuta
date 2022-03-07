import pkg_resources
from .utils import load_sigs, load_config

def load_cosmic_V3():
    """Return a dataframe of COSMIC V3 signature definitions

    Contains:
        96 mutation type columns of non-null float64
        78 rows of signature definitions, rows sum to 1

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    f = pkg_resources.resource_filename(__name__, 'data/COSMIC_v3.2_SBS_GRCh37.txt')
    return load_sigs(f)

def load_default_config():
    """Return a default configuration dict

    Contains:
        96 mutation type columns of non-null float64
        78 rows of signature definitions, rows sum to 1

    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    f = pkg_resources.resource_filename(__name__, 'config/default.yaml')
    return load_config(f)
