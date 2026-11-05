import numpy as np
import pytest
from scripts.detect_onsets import compare, group_onsets, reference_onsets


def test_reference_preserves_attacks_under_sustained_other_string():
    tab = np.zeros((10, 6))
    tab[1:9, 0] = 1
    tab[3:5, 1] = 1
    tab[7:9, 1:3] = 1
    np.testing.assert_allclose(reference_onsets(tab), np.array([1,3,7])/44)


def test_matching_is_one_to_one_and_keeps_boundary_matches():
    result = compare([1, 1.04, 2], [1.02, 2.05, 3], .051)
    assert result['matches'] == 2
    assert result['false_positives'] == 1
    assert result['misses'] == 1
    assert result['f1'] == pytest.approx(2/3)


def test_groups_do_not_chain_across_entire_phrase():
    np.testing.assert_allclose(group_onsets([0,.06,.12,.18],.1),[0,.12])


def test_empty_comparison():
    assert compare([],[],.05)['f1'] == 0


def test_invalid_reference():
    with pytest.raises(ValueError):
        reference_onsets(np.zeros((10,5)))
