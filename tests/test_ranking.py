import math

import pytest

from mmrr.data.schema import Sequence
from mmrr.eval.ranking import candidate_record, rank_of, rerank_record, sort_by_score


def test_rank_is_1_based_and_none_when_missing():
    assert rank_of([7, 8, 9], 7) == 1
    assert rank_of([7, 8, 9], 9) == 3
    assert rank_of([7, 8, 9], 1) is None


def test_sort_desc_and_ties_keep_original_order():
    items, scores = sort_by_score([10, 11, 12, 13], [0.5, 2.0, 0.5, -1.0])
    assert items == [11, 10, 12, 13]
    assert scores == [2.0, 0.5, 0.5, -1.0]


def test_sort_rejects_nan_and_length_mismatch():
    with pytest.raises(ValueError):
        sort_by_score([1, 2], [0.1, math.nan])
    with pytest.raises(ValueError):
        sort_by_score([1, 2], [0.1])


def test_candidate_record_fields():
    seq = Sequence(user="u1", history=[1, 2, 3, 4, 5], target=9)
    r = candidate_record(seq, [8, 9, 7], [0.9, 0.8, 0.1])
    assert r == {"user": "u1", "history": [1, 2, 3, 4, 5], "target": 9,
                 "candidates": [8, 9, 7], "scores": [0.9, 0.8, 0.1], "target_rank": 2}
    assert candidate_record(seq, [8, 7], [0.9, 0.1])["target_rank"] is None


def test_candidate_record_rejects_unsorted_and_history_leak():
    seq = Sequence(user="u1", history=[1, 2], target=9)
    with pytest.raises(ValueError):
        candidate_record(seq, [8, 9], [0.1, 0.9])
    with pytest.raises(ValueError):
        candidate_record(seq, [8, 1], [0.9, 0.1])


def test_rerank_record_keeps_retrieval_order_and_updates_rank():
    cand = {"user": "u1", "history": [1], "target": 9,
            "candidates": [8, 9, 7], "scores": [0.9, 0.8, 0.1], "target_rank": 2}
    r = rerank_record(cand, [-1.0, 3.0, 0.0])
    assert r["candidates"] == [8, 9, 7]
    assert r["scores"] == [-1.0, 3.0, 0.0]
    assert r["reranked"] == [9, 7, 8]
    assert r["target_rank_retrieval"] == 2
    assert r["target_rank_rerank"] == 1


def test_rerank_record_target_not_retrieved_stays_none():
    cand = {"user": "u1", "history": [1], "target": 5,
            "candidates": [8, 9], "scores": [0.9, 0.8], "target_rank": None}
    r = rerank_record(cand, [0.0, 1.0])
    assert r["target_rank_retrieval"] is None and r["target_rank_rerank"] is None
