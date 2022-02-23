import pytest
from damuta import load_sigs, detect_naming_style

def test_correct_naming():
    assert detect_naming_style('data/degasperi_localsigs.csv') == 'type'
    assert detect_naming_style('data/degasperi_refsigs.csv') == 'type/subtype'

def test_incorrect_naming():
    with pytest.raises(AssertionError):
        detect_naming_style('data/icgc_sample_annotations_summary_table.txt')


@pytest.mark.parametrize(
    "sig_fp",
    ['data/COSMIC_v3.2_SBS_GRCh37.txt',
     'data/degasperi_localsigs.csv',
     'data/degasperi_refsigs.csv',
     'data/pcawg_localsigs.csv',
    ]
)
def test_provided_sigs_loadable(sig_fp):
    sig_defs = load_sigs(sig_fp)
    assert sig_defs.shape[0] > 0, 'No signatures loaded'
    assert sig_defs.shape[1] == 96, 'Incorrect nmber of mutation types'

     