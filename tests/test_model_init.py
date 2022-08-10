import pytest
from damuta.models import Lda, TandemLda, HierarchicalTandemLda

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
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='tissue_type', init_strategy='uniform')
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
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10,n_misrepair_sigs=5, type_col='tissue_type', init_strategy='kmeans')
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
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=78, n_misrepair_sigs=78, type_col='tissue_type', init_strategy='from_sigs', init_signatures=cosmic)
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
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='tissue_type', init_strategy='uniform', init_signatures=cosmic)
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
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='tissue_type', init_strategy='kmeans', init_signatures=cosmic)
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
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=78, type_col='tissue_type', init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
    with pytest.warns(UserWarning, match ='init_signatures misrepair dimension does not match n_misrepair_sigs of 5. Argument n_misrepair_sigs will be ignored.'):
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=78, n_misrepair_sigs=5, type_col='tissue_type', init_strategy='from_sigs', init_signatures=cosmic)
        model.fit(2)
        assert model.n_damage_sigs == 78
        assert model.n_misrepair_sigs == 78
    with pytest.warns(UserWarning) as record:
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='tissue_type', init_strategy='from_sigs', init_signatures=cosmic)
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
        model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='tissue_type', init_strategy='from_sigs')
        model.fit(2)

