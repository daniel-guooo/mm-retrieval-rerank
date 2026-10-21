from collections import Counter
import pytest
from mmrr.multiroute import quota_union


def test_duplicates_backfilled_with_exact_quota():
    items, scores, sources = quota_union([[1, 2, 3, 4, 5], [1, 2, 4, 3, 5], [4, 1, 5, 3, 2]], [2, 1, 1])
    assert set(items) == {1, 2, 4, 5}
    assert Counter(sources) == {0: 2, 1: 1, 2: 1}
    assert scores == sorted(scores, reverse=True)
    assert dict(zip(items, sources)) == {1: 0, 2: 0, 4: 1, 5: 2}


def test_insufficient_depth_and_zero_quota():
    with pytest.raises(ValueError):
        quota_union([[1, 2], [1, 2]], [2, 1])
    assert quota_union([[1], [2]], [0, 1])[0] == [2]


def test_equal_rrf_tie_is_item_id_order():
    assert quota_union([[2, 1], [1, 2]], [1, 1])[0] == [1, 2]
