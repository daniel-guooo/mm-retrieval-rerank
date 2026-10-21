"""Exact full-catalog retrieval on cached vectors (CPU, no model loading)."""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

METHODS = ("popularity", "text", "image", "fusion")


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def item_vectors(text, image, method, alpha=0.5):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if method == "text":
        return text
    if method == "image":
        return image
    if method == "fusion":
        return normalize(alpha * text + (1 - alpha) * image)
    raise ValueError(method)


def popularity_scores(item_ids, train_rows):
    counts = Counter(i for _, history in train_rows for i in history)
    return np.asarray([counts[int(i)] for i in item_ids], dtype=np.float64)


def retrieve(sequences, item_ids, vectors=None, popularity=None, gamma=1.0, k=100, batch_size=128):
    """Yields (sequence, candidate IDs, scores). Target is never used for scoring.

    gamma=1 gives equal history weights; gamma<1 discounts older interactions.
    Ties are resolved by ascending original item ID. Exclusion uses input history.
    """
    if not 0 < gamma <= 1 or k < 1 or batch_size < 1:
        raise ValueError("Require 0 < gamma <= 1, k >= 1, batch_size >= 1")
    if (vectors is None) == (popularity is None):
        raise ValueError("Provide exactly one of vectors or popularity")
    if len(np.unique(item_ids)) != len(item_ids) or np.any(item_ids[1:] <= item_ids[:-1]):
        raise ValueError("Item IDs must be unique and ascending")
    lookup = {int(i): row for row, i in enumerate(item_ids)}
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        histories = []
        for seq in chunk:
            if not seq.history:
                raise ValueError(f"Empty history: {seq.user}")
            histories.append([lookup[i] for i in seq.history])
        if vectors is not None:
            users = []
            for rows in histories:
                weights = np.power(gamma, np.arange(len(rows) - 1, -1, -1)).astype(np.float32)
                users.append(normalize((vectors[rows] * weights[:, None]).sum(axis=0)))
            scores = np.asarray(users, dtype=np.float32) @ vectors.T
        else:
            scores = np.broadcast_to(popularity, (len(chunk), len(item_ids))).copy()
        for seq, hist, row in zip(chunk, histories, scores):
            row[hist] = -np.inf
            order = np.argsort(-row, kind="stable")[:k]
            order = order[np.isfinite(row[order])]
            yield seq, item_ids[order].tolist(), row[order].tolist()


def summarize(ranks, candidates_at_50, n_items):
    if not ranks:
        raise ValueError("No evaluation queries")
    n = len(ranks)
    result = {f"Recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in (10, 20, 50, 100)}
    for k in (3, 10):
        result[f"HR@{k}"] = sum(r is not None and r <= k for r in ranks) / n
        result[f"NDCG@{k}"] = sum(1 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / n
    result["coverage@50"] = len(candidates_at_50) / n_items
    return result
