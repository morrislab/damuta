from damuta.models import HierarchicalTandemLda
from damuta.base import DataSet

def test_ADVI(pcawg):
    
    model = HierarchicalTandemLda(dataset=pcawg, n_damage_sigs=10, 
                                  n_misrepair_sigs=5, type_col='tissue_type',
                                  opt_method = "ADVI")
    model.fit(15)
    
def test_FullRankADVI(pcawg):
    pcawg_subset = DataSet(pcawg.counts[0:100], pcawg.annotation[0:100])
    model = HierarchicalTandemLda(dataset=pcawg_subset, n_damage_sigs=4, 
                                  n_misrepair_sigs=3, type_col='tissue_type',
                                  opt_method = "FullRankADVI")
    model.fit(15)