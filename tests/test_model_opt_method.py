import pytest
from damuta.models import Lda, TandemLda, HierarchicalTandemLda

@pytest.mark.slow  
def test_ADVI(pcawg):
    model = Lda(dataset=pcawg, n_sigs=10,  opt_method = "ADVI")
    model.fit(2)
    model = TandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, opt_method = "ADVI")
    model.fit(2)
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, n_misrepair_sigs=5, type_col='pcawg_class', opt_method = "ADVI")
    model.fit(2)

@pytest.mark.slow      
def test_FullRankADVI(pcawg):
    model = Lda(dataset=pcawg, n_sigs=10,  opt_method = "FullRankADVI")
    model.fit(2)
    model = TandemLda(dataset=pcawg, n_damage_sigs=4, n_misrepair_sigs=3, opt_method = "FullRankADVI")
    model.fit(2)
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=4, n_misrepair_sigs=3, type_col='pcawg_class', opt_method = "FullRankADVI")
    model.fit(2)