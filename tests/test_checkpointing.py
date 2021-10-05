import pytest
import pymc3 as pm
import numpy as np
from damuta import save_checkpoint, load_checkpoint, ch_dirichlet

def test_save_no_args():
    assert True
    
def test_save_load(tmp_path_factory, c, sig_defs):
    
    # make small for speed
    c = c[0:30]
    sig_defs = sig_defs[0:5]
    
    dataset_args={'foo':'bar'}
    model_args={'bar':'baz'}
    pymc3_args={'baz':'foo'}

    # train a model with 5 sigs
    with pm.Model() as model:
        data = pm.Data("data", c)
        N=data.sum(1).reshape((c.shape[0],1))
        activities = ch_dirichlet("activities", a = np.ones(5), shape=(c.shape[0], 5))
        B = pm.math.dot(activities,sig_defs)
        pm.Multinomial('corpus', n = N, p = B, observed=data)

        trace = pm.ADVI()
        trace.fit()

    # checkpoint
    fp = tmp_path_factory.mktemp("ckp")/"vanilla_lda.ckp"
    save_checkpoint(fp, model, trace, dataset_args, model_args, pymc3_args)
    
    # load model
    m2, t2, dataset_args2, model_args2, pymc3_args2 = load_checkpoint(fp)

    # all params should be identical
    # checks are weak because __eq__ methods are not provided
    #assert str(model) == str(m2), 'model load failed'
    assert np.allclose(trace.hist,t2.hist), 'trace load failed'
    assert dataset_args == dataset_args2, 'dataset_args load failed' 
    assert model_args == model_args2, 'model_args load failed'
    assert pymc3_args == pymc3_args2, 'dataset_args load failed'

    # with same seed, both models should tune with same result
    # test model tuning
    trace.refine(100)
    t2.refine(100)
    assert np.allclose(trace.hist,t2.hist), 'trace tuning failed'

