from damuta.models import Lda, TandemLda, HierarchicalTandemLda

def test_DataSet_init(pcawg):
    assert pcawg.n_samples == 2778
    
def test_SignatureSet_init(cosmic):
    assert cosmic.n_sigs == 78
    assert cosmic.damage_signatures.shape == (78,32)
    assert cosmic.misrepair_signatures.shape == (78,6)
    assert cosmic.summarize_separation().shape == (8,3)
    
def test_Lda(pcawg):
    
    model = Lda(dataset=pcawg, n_sigs=10)
    assert model.dataset.counts.shape == pcawg.counts.shape
    model.fit(15)

def test_TandemLda(pcawg):
    
    model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5)
    assert model.dataset.counts.shape == pcawg.counts.shape
    model.fit(15)
    
def test_HierarchicalTandemLda(pcawg):
    
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, 
                                  n_misrepair_sigs=5, type_col='tissue_type')
    assert model.dataset.counts.shape == pcawg.counts.shape
    model.fit(15)
