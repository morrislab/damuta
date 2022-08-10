import pandas as pd
from damuta.utils import get_phi, get_eta, mut32, mut6
from damuta.base import SignatureSet

def test_DataSet_init(pcawg):
    assert pcawg.n_samples == 100
    
def test_SignatureSet_init(cosmic):
    assert cosmic.n_sigs == 78
    assert cosmic.damage_signatures.shape == (78,32)
    assert cosmic.misrepair_signatures.shape == (78,6)
    assert cosmic.summarize_separation().shape == (8,3)

def test_SignatureSet_init_from_signatures(cosmic):
    from_tau = SignatureSet(cosmic.signatures)
    assert from_tau.n_sigs == 78
    assert from_tau.n_damage_sigs == 78
    assert from_tau.n_misrepair_sigs == 78

def test_SignatureSet_init_from_damage_misrepair(cosmic):
    phi = pd.DataFrame(get_phi(cosmic.signatures.to_numpy()), index = cosmic.signatures.index, columns = mut32)[0:10]
    eta = pd.DataFrame(get_eta(cosmic.signatures.to_numpy()), index = cosmic.signatures.index, columns = mut6)[0:5]
    from_phi_eta = SignatureSet.from_damage_misrepair(phi, eta)
    assert from_phi_eta.n_sigs == 50
    assert from_phi_eta.n_damage_sigs == 10
    assert from_phi_eta.n_misrepair_sigs == 5
    
