"""候选文件 / 重排文件的记录格式，以及排序约定。

约定：
- 名次从 1 开始；target 不在列表里时为 None（写进 json 是 null）
- 分数越大越靠前；分数相同保持原顺序（稳定排序）
- 不会把 target 手动塞进候选
"""
from __future__ import annotations

import math
from typing import Sequence as Seq

from mmrr.data.schema import Sequence


def rank_of(items: Seq[int], target: int) -> int | None:
    for i, x in enumerate(items):
        if x == target:
            return i + 1
    return None


def sort_by_score(items: Seq[int], scores: Seq[float]) -> tuple[list[int], list[float]]:
    if len(items) != len(scores):
        raise ValueError(f"长度不一致: {len(items)} vs {len(scores)}")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("分数里有 NaN/inf")
    order = sorted(range(len(items)), key=lambda i: -scores[i])  # Python sort 是稳定的
    return [int(items[i]) for i in order], [float(scores[i]) for i in order]


def candidate_record(seq: Sequence, candidates: Seq[int], scores: Seq[float]) -> dict:
    """manifests/<dataset>/<split>_top50.jsonl 的一行。candidates 必须已按分数降序。"""
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        raise ValueError("candidates 必须按分数降序")
    if set(candidates) & set(seq.history):
        raise ValueError(f"user {seq.user}: 候选里出现了历史商品")
    return {
        "user": seq.user,
        "history": list(seq.history),
        "target": seq.target,
        "candidates": [int(c) for c in candidates],
        "scores": [round(float(s), 6) for s in scores],
        "target_rank": rank_of(candidates, seq.target),
    }


def rerank_record(cand: dict, rerank_scores: Seq[float]) -> dict:
    """outputs/rerank/<run>/<dataset>/<split>.jsonl 的一行。

    candidates / scores 保持召回顺序（和候选文件逐位对应），reranked 是重排后的结果。
    """
    reranked, _ = sort_by_score(cand["candidates"], rerank_scores)
    return {
        "user": cand["user"],
        "target": cand["target"],
        "candidates": cand["candidates"],
        "scores": [round(float(s), 6) for s in rerank_scores],
        "reranked": reranked,
        "target_rank_retrieval": cand["target_rank"],
        "target_rank_rerank": rank_of(reranked, cand["target"]),
    }
