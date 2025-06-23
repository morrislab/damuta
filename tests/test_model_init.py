import pytest
from damuta.models import Lda, TandemLda, HierarchicalTandemLda
import numpy as np

## Models should build with no errors
@pytest.mark.slow
def test_Lda_init_uniform_build(pcawg):
    model = Lda(dataset=pcawg, n_sigs=10, init_strategy='uniform')
    model.fit(2)

@pytest.mark.slow  
def test_TandemLda_init_uniform_build(pcawg):
    model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, init_strategy='uniform')
    model.fit(2)

@pytest.mark.slow
def test_HierarchicalTandemLda_init_uniform_build(pcawg):
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='uniform')
    model.fit(2)

@pytest.mark.slow
def test_Lda_init_kmeans_build(pcawg):
    model = Lda(dataset=pcawg, n_sigs=10, init_strategy='kmeans')
    model.fit(2)

@pytest.mark.slow   
def test_TandemLda_init_kmeans_build(pcawg):
    model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, init_strategy='kmeans')
    model.fit(2)
    
@pytest.mark.slow  
def test_HierarchicalTandemLda_init_kmeans_build(pcawg):
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10,n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='kmeans')
    model.fit(2)

@pytest.mark.slow  
def test_Lda_init_from_sigs_build(pcawg, cosmic):
    model = Lda(dataset=pcawg, n_sigs=78, init_strategy='from_sigs', init_signatures=cosmic)
    model.fit(2)

@pytest.mark.slow   
def test_TandemLda_init_from_sigs_build(pcawg, cosmic):
    model = TandemLda(dataset=pcawg, n_damage_sigs=78, n_misrepair_sigs=78, init_strategy='from_sigs', init_signatures=cosmic)
    model.fit(2)

@pytest.mark.slow  
def test_HierarchicalTandemLda_init_from_sigs_build(pcawg, cosmic):
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=78, n_misrepair_sigs=78, type_col='pcawg_class', init_strategy='from_sigs', init_signatures=cosmic)
    model.fit(2)

## Models should handle init_signatures conflicts

@pytest.mark.slow  
def test_init_signatures_and_uniform_throws_warning(pcawg, cosmic):
    with pytest.warns(UserWarning, match ='signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.'):
        model = Lda(dataset=pcawg, n_sigs=10, init_strategy='uniform', init_signatures=cosmic)
        model.fit(2)
    with pytest.warns(UserWarning, match ='signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.'):
        model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, init_strategy='uniform', init_signatures=cosmic)
        model.fit(2)
    with pytest.warns(UserWarning, match ='signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.'):
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='uniform', init_signatures=cosmic)
        model.fit(2)
        
@pytest.mark.slow  
def test_init_signatures_and_kmeans_throws_warning(pcawg, cosmic):
    with pytest.warns(UserWarning, match ='signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.'):
        model = Lda(dataset=pcawg, n_sigs=10, init_strategy='kmeans', init_signatures=cosmic)
        model.fit(2)
    with pytest.warns(UserWarning, match ='signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.'):
        model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, init_strategy='kmeans', init_signatures=cosmic)
        model.fit(2)
    with pytest.warns(UserWarning, match ='signature_set provided, but init_strategy is not "from_sigs". signature_set will be ignored.'):
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='kmeans', init_signatures=cosmic)
        model.fit(2)

@pytest.mark.slow  
def test_Lda_init_signatures_and_bad_n_sigs_throws_warning(pcawg, cosmic):
    with pytest.warns(UserWarning, match = 'init_signatures signature dimension does not match n_sigs of 10. Argument n_sigs will be ignored.'):
        model = Lda(dataset=pcawg, n_sigs=10, init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_sigs == 78

@pytest.mark.slow  
def test_TandemLda_init_signatures_and_bad_n_sigs_throws_warning(pcawg, cosmic):
    with pytest.warns(UserWarning, match ='init_signatures damage dimension does not match n_damage_sigs of 10. Argument n_damage_sigs will be ignored.'):
        model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=78, init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
    with pytest.warns(UserWarning, match ='init_signatures misrepair dimension does not match n_misrepair_sigs of 5. Argument n_misrepair_sigs will be ignored.'):
        model = TandemLda(dataset=pcawg, n_damage_sigs=78, n_misrepair_sigs=5, init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
    with pytest.warns(UserWarning) as record:
        model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
        assert 'Argument n_damage_sigs will be ignored.' in str(record[0].message)
        assert 'Argument n_misrepair_sigs will be ignored.' in str(record[1].message) 

@pytest.mark.slow  
def test_HierarchicalTandemLda_init_signatures_and_bad_n_sigs_throws_warning(pcawg, cosmic):
    with pytest.warns(UserWarning, match ='init_signatures damage dimension does not match n_damage_sigs of 10. Argument n_damage_sigs will be ignored.'):
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=78, type_col='pcawg_class', init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
    with pytest.warns(UserWarning, match ='init_signatures misrepair dimension does not match n_misrepair_sigs of 5. Argument n_misrepair_sigs will be ignored.'):
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=78, n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
    with pytest.warns(UserWarning) as record:
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
        assert 'Argument n_damage_sigs will be ignored.' in str(record[0].message)
        assert 'Argument n_misrepair_sigs will be ignored.' in str(record[1].message) 

@pytest.mark.slow  
def test_init_signatures_and_no_sigs_throws_error(pcawg):
    with pytest.raises(AssertionError, match = 'init_strategy "from_sigs" requires a signature set to be passed.'):
        model = Lda(dataset=pcawg, n_sigs=10, init_strategy='from_sigs')
        model.fit(2)
    with pytest.raises(AssertionError, match ='init_strategy "from_sigs" requires a signature set to be passed.'):
        model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, init_strategy='from_sigs')
        model.fit(2)
    with pytest.raises(AssertionError, match ='init_strategy "from_sigs" requires a signature set to be passed.'):
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='pcawg_class', init_strategy='from_sigs')
        model.fit(2)

def create_dummy_dataset():
    import pandas as pd
    from damuta.base import DataSet
    from damuta.constants import mut96
    # Create proper DataFrame with mutation type columns  
    counts = pd.DataFrame(np.zeros((2, 96)), columns=mut96)
    counts.index = ["sample_0", "sample_1"]
    annotation = pd.DataFrame({'type': ['A', 'B']}, index=counts.index)
    return DataSet(counts, annotation)

def test_tandemlda_n_damage_sigs_required_if_phi_obs_none():
    with pytest.raises(ValueError, match="n_damage_sigs must be provided when phi_obs is None"):
        TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=None, n_misrepair_sigs=2, phi_obs=None, etaC_obs=None, etaT_obs=None)

def test_tandemlda_n_damage_sigs_ignored_if_phi_obs_provided():
    import warnings
    phi_obs = np.zeros((3, 32))
    with warnings.catch_warnings(record=True) as w:
        TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=2, n_misrepair_sigs=2, phi_obs=phi_obs, etaC_obs=None, etaT_obs=None)
        assert any("argument n_damage_sigs" in str(x.message) for x in w)

def test_tandemlda_n_damage_sigs_set_from_phi_obs_if_none():
    phi_obs = np.zeros((4, 32))
    model = TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=None, n_misrepair_sigs=2, phi_obs=phi_obs, etaC_obs=None, etaT_obs=None)
    assert model.n_damage_sigs == 4

def test_tandemlda_etaC_and_etaT_must_be_provided_together():
    etaC_obs = np.zeros((2, 3))
    with pytest.raises(ValueError, match="etaC_obs and etaT_obs must be provided together"):
        TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=2, n_misrepair_sigs=2, phi_obs=None, etaC_obs=etaC_obs, etaT_obs=None)

def test_tandemlda_n_misrepair_sigs_required_if_eta_obs_none():
    with pytest.raises(ValueError, match="n_misrepair_sigs must be provided when etaC_obs and etaT_obs are None"):
        TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=2, n_misrepair_sigs=None, phi_obs=None, etaC_obs=None, etaT_obs=None)

def test_tandemlda_etaC_and_etaT_shape_must_match():
    etaC_obs = np.zeros((2, 3))
    etaT_obs = np.zeros((3, 3))
    with pytest.raises(ValueError, match="etaC_obs.shape\[0\] \(2\) must equal etaT_obs.shape\[0\] \(3\)"):
        TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=2, n_misrepair_sigs=2, phi_obs=None, etaC_obs=etaC_obs, etaT_obs=etaT_obs)

def test_tandemlda_n_misrepair_sigs_ignored_if_eta_obs_provided():
    import warnings
    etaC_obs = np.zeros((5, 3))
    etaT_obs = np.zeros((5, 3))
    with warnings.catch_warnings(record=True) as w:
        TandemLda(dataset=create_dummy_dataset(), n_damage_sigs=2, n_misrepair_sigs=2, phi_obs=None, etaC_obs=etaC_obs, etaT_obs=etaT_obs)
        assert any("argument n_misrepair_sigs" in str(x.message) for x in w)

