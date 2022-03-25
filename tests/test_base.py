import pandas as pd
from damuta.models import Lda

def test_DataSet_init(pcawg):
    assert pcawg.nsamples == 2778
    
def test_SignatureSet_init(cosmic):
    assert cosmic.nsigs == 78
    assert cosmic.damage_signatures.shape == (78,32)
    assert cosmic.misrepair_signatures.shape == (78,6)
    assert cosmic.summarize_separation.shape == (8,3)
    
def test_Lda_init(pcawg):
    
    model = Lda(pcawg)
    
    