import pytest
from damut import detect_naming_style

def test_correct_naming():
    assert detect_naming_style('data/degasperi_localsigs.csv') == 'type'
    assert detect_naming_style('data/degasperi_refsigs.csv') == 'type/subtype'

def test_incorrect_naming():
    with pytest.raises(AssertionError):
        detect_naming_style('data/icgc_sample_annotations_summary_table.txt')

