import pytest
import pandas as pd
from numpy.random import default_rng
from damut import subset_samples, split_by_count, split_by_S

c = pd.read_csv('data/mutation_types_raw_counts.csv', index_col=0, header=0)
a = pd.read_csv('data/pcawg_cancer_types.csv', index_col=0, header=0)

def test_no_subsetting():
    assert all(c == subset_samples(c,a,None))
    
def test_list_equiv():
    assert all(subset_samples(c,a,'Breast') == subset_samples(c,a,['Breast'])), 'Sring processing failed'

@pytest.mark.parametrize(
    "subset,sel,expected,c,a",
    [#0: len 1 list match
     (['Breast-AdenoCA'], a['type'] == 'Breast-AdenoCA', c.loc[ a.index[a['type'] == 'Breast-AdenoCA'] ], c, a), 
     #1: full string match
     ('Breast-AdenoCA', a['type'] == 'Breast-AdenoCA', c.loc[ a.index[a['type'] == 'Breast-AdenoCA'] ], c, a),
     #2: partial string match with one hit
     ('Colo', a['type'] == 'ColoRect-AdenoCA', c.loc[ a.index[a['type'] == 'ColoRect-AdenoCA'] ], c, a),
     #3: partial string match with multiple hits
     ('Kidney', a['type'].isin(['Kidney-RCC', 'Kidney-ChRCC']), c.loc[ a.index[a['type'].isin(['Kidney-RCC', 'Kidney-ChRCC'])] ], c, a),
     #4: len >1 list
     (['Skin', 'Colo'], a['type'].isin(['Skin-Melanoma', 'ColoRect-AdenoCA']), c.loc[ a.index[a['type'].isin(['Skin-Melanoma', 'ColoRect-AdenoCA'])] ], c, a),
     #5: disordered
     ('Breast-AdenoCA', a['type'] == 'Breast-AdenoCA', c.sample(frac=1).loc[ a.index[a['type'] == 'Breast-AdenoCA'] ], c, a),
     #6: more annotations than data samples
     ('Ovary-AdenoCA', pd.Series([True] + [False] * (c.shape[0]-1), index=c.index), c[0:1], c[0:1], a), 
     #7: more datasamples than annotations
     ('Ovary-AdenoCA', pd.Series([True, False, False], index=c.index[0:3]), c[0:1], c[0:3], a[0:1]) 
    ]
)
def test_subsetting(subset, sel, expected, c, a):
    c_subsetted = subset_samples(c,a,subset)
    assert c_subsetted.shape == (sel.sum(), 96), 'count shape fail'
    assert all(c_subsetted.index.isin(sel.index[sel])), 'index build fail'
    assert all(c_subsetted == expected), 'subsetting fail'
    
def test_no_match():
    with pytest.raises(AssertionError):
        subset_samples(c,a,'zilch')
    assert all(subset_samples(c,a,['Breast','zilch']) == subset_samples(c,a,['Breast'])), 'No match fails in combo'

def test_data_splitting():
    split_by_count(c, 0.8, rng=default_rng())
    split_by_S(c, 0.8, rng=default_rng())