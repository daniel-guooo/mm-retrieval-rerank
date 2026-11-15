"""Evaluation against a complete natural manifest, including unscored misses."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from mmrr.eval.ranking import rank_of, sort_by_score

RANK_METRICS = ("HR@3", "HR@10", "NDCG@3", "NDCG@10")


def _index(rows, *, allow_empty=False):
    users = {}
    for row in rows:
        if row["user"] in users:
            raise ValueError(f"Duplicate user: {row['user']}")
        users[row["user"]] = row
    if not users and not allow_empty:
        raise ValueError("Empty ranking file")
    return users


def natural_rows(manifest):
    users = _index(manifest)
    for row in users.values():
        items = row["candidates"]
        if (row.get("oracle_injected") or not items or len(set(items)) != len(items)
                or set(items).intersection(row["history"])
                or row["target_rank"] != rank_of(items, row["target"])):
            raise ValueError(f"Invalid natural candidate manifest for {row['user']}")
    return users


def match_scores(users, scored):
    """Check membership and ensure all natural hits have scores."""
    indexed = _index(scored, allow_empty=True)
    if indexed.keys() - users.keys():
        raise ValueError("Scores contain users outside the manifest")
    for user, row in users.items():
        result = indexed.get(user)
        if result is None:
            if row["target_rank"] is not None:
                raise ValueError(f"Missing scores for a natural hit: {user}")
            continue
        items = row["candidates"]
        if (result["candidates"] != items or result["target"] != row["target"]
                or len(result["scores"]) != len(items) or not all(map(math.isfinite, result["scores"]))
                or set(result["reranked"]) != set(items) or len(result["reranked"]) != len(items)):
            raise ValueError(f"Scores do not match the natural candidates for {user}")
    return indexed


def _metric(rank, name):
    value, cutoff = name.split("@")
    if rank is None or rank > int(cutoff):
        return 0.0
    return 1.0 if value == "HR" else 1.0 / math.log2(rank + 1)


def summarize_ranks(ranks):
    return {name: sum(_metric(rank, name) for rank in ranks) / len(ranks) if ranks else None
            for name in RANK_METRICS}


def evaluate_natural(manifest, scored=None, *, k=None, n_items=None, popularity=None):
    users = natural_rows(manifest)
    results = match_scores(users, scored) if scored is not None else {}
    per_user = []
    coverage = set()
    budgets = {len(row["candidates"]) for row in users.values()}
    if len(budgets) != 1 or (k is not None and not 1 <= k <= next(iter(budgets))):
        raise ValueError("Use a consistent candidate budget")
    k = k or next(iter(budgets))
    for user, row in users.items():
        items = row["candidates"][:k]
        coverage.update(items)
        original_rank = rank_of(items, row["target"])
        ranked = results[user]["reranked"] if user in results else items
        selected = set(items)
        rank = rank_of([item for item in ranked if item in selected], row["target"])
        per_user.append({"user": user, "target": row["target"], "retrieval_rank": original_rank,
                         "rank": rank})
    hits = [row for row in per_user if row["retrieval_rank"] is not None]
    summary = {
        "n_users": len(per_user), "n_scored": len(results), "n_hits": len(hits), "k": k,
        "candidate_recall": len(hits) / len(per_user), "catalog_items_retrieved": len(coverage),
        "catalog_coverage": len(coverage) / n_items if n_items else None,
        "retrieval": summarize_ranks([row["retrieval_rank"] for row in per_user]),
        "conditional_retrieval": summarize_ranks([row["retrieval_rank"] for row in hits]),
        "ranking": summarize_ranks([row["rank"] for row in per_user]),
        "conditional_ranking": summarize_ranks([row["rank"] for row in hits]),
        "recall": {f"Recall@{cutoff}": sum(_metric(row["retrieval_rank"], f"HR@{cutoff}")
                                         for row in per_user) / len(per_user)
                   for cutoff in (10, 20, 50, 100) if cutoff <= k},
    }
    if popularity is not None:
        buckets = Counter()
        bucket_hits = Counter()
        for row in per_user:
            count = popularity.get(row["target"], 0)
            bucket = "0" if count == 0 else "1-10" if count <= 10 else "11-100" if count <= 100 else "101+"
            buckets[bucket] += 1
            bucket_hits[bucket] += row["retrieval_rank"] is not None
        summary["recall_by_train_frequency"] = {
            bucket: {"n_users": n, "hits": bucket_hits[bucket], "recall": bucket_hits[bucket] / n}
            for bucket, n in sorted(buckets.items())
        }
    return summary, per_user


def fused_runs(manifest, scored, weight):
    """Combine rank percentiles, preserving archived Qwen and CLIP endpoints."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Fusion weight must be in [0, 1]")
    users = natural_rows(manifest)
    results = match_scores(users, scored)
    for user, result in results.items():
        items = users[user]["candidates"]
        qwen_position = {i: pos for pos, i in enumerate(result["reranked"])}
        scale = max(1, len(items) - 1)
        scores = [1 - ((1 - weight) * pos + weight * qwen_position[item]) / scale
                  for pos, item in enumerate(items)]
        ranked, _ = sort_by_score(items, scores)
        yield {**result, "scores": scores, "reranked": ranked,
               "target_rank_rerank": rank_of(ranked, result["target"])}


def paired_bootstrap(baseline, variant, *, samples=2000, seed=2025):
    before, after = _index(baseline), _index(variant)
    if before.keys() != after.keys() or samples < 1:
        raise ValueError("Both runs must have the same users and a positive bootstrap budget")
    if any((before[user]["target"], before[user]["retrieval_rank"])
           != (after[user]["target"], after[user]["retrieval_rank"]) for user in before):
        raise ValueError("Mismatched evaluation targets or retrieval ranks")
    delta = np.array([[_metric(after[user]["rank"], name) - _metric(before[user]["rank"], name)
                       for name in RANK_METRICS] for user in before], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 64):
        indexes = rng.integers(0, len(delta), (min(64, samples - start), len(delta)))
        draws.append(delta[indexes].mean(axis=1))
    low, high = np.quantile(np.concatenate(draws), [0.025, 0.975], axis=0)
    return {name: {"difference": float(mean), "ci95": [float(a), float(b)]}
            for name, mean, a, b in zip(RANK_METRICS, delta.mean(axis=0), low, high, strict=True)}
