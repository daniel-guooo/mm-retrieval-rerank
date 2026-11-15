"""Candidate scores for distinct history interests; history rows are oldest first."""
import numpy as np

METHODS = ('last', 'last2_mean', 'history_max', 'history_top2_mean',
           'max_last_025', 'max_last_050', 'max_last_075')


def interest_scores(similarities, method):
    if similarities.ndim != 2 or len(similarities) < 1:
        raise ValueError('Expected nonempty history x catalog similarities')
    last = similarities[-1]
    if method == 'last':
        return last.copy()
    if method == 'last2_mean':
        return similarities[-2:].mean(axis=0)
    maximum = similarities.max(axis=0)
    if method == 'history_max':
        return maximum
    if method == 'history_top2_mean':
        return np.sort(similarities, axis=0)[-2:].mean(axis=0)
    if method in ('max_last_025', 'max_last_050', 'max_last_075'):
        last_weight = {'max_last_025': .25, 'max_last_050': .5, 'max_last_075': .75}[method]
        return last_weight*last + (1-last_weight)*maximum
    raise ValueError(method)


def history_similarities(sequences, item_ids, vectors, batch_size=128, *, lookup=None):
    """Yield each query's history-by-catalog similarities and history row indices."""
    if batch_size < 1 or len(vectors) != len(item_ids) or np.any(item_ids[1:] <= item_ids[:-1]):
        raise ValueError("Use a positive batch size and vectors aligned to unique ascending IDs")
    if lookup is None:
        lookup = {int(item): row for row, item in enumerate(item_ids)}
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start:start + batch_size]
        histories = [[lookup[item] for item in seq.history] for seq in chunk]
        flat = [row for history in histories for row in history]
        similarities = vectors[flat] @ vectors.T
        offset = 0
        for seq, history in zip(chunk, histories, strict=True):
            stop = offset + len(history)
            yield seq, history, similarities[offset:stop]
            offset = stop


def retrieve_interest(sequences, item_ids, vectors, method="max_last_025", k=100, batch_size=128, *, lookup=None):
    """Exact search with the same exclusion and tie rules as the frozen manifests."""
    if k < 1 or k > len(item_ids):
        raise ValueError("K must be positive and at most the catalog size")
    for seq, history, similarities in history_similarities(sequences, item_ids, vectors, batch_size, lookup=lookup):
        scores = interest_scores(similarities, method)
        scores[history] = -np.inf
        order = np.argsort(-scores, kind="stable")[:k]
        order = order[np.isfinite(scores[order])]
        yield seq, item_ids[order].tolist(), scores[order].tolist()


class FaissInterestSearch:
    """Shortlist neighbors per history item and rescore their union with the interest formula."""

    def __init__(self, item_ids, vectors, depth):
        import faiss

        self.item_ids = item_ids
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.depth = min(depth, len(item_ids))
        self.lookup = {int(item): row for row, item in enumerate(item_ids)}
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        self.index.add(self.vectors)

    def retrieve(self, seq, k):
        history = [self.lookup[item] for item in seq.history]
        if self.depth < k + len(set(history)):
            raise ValueError("FAISS shortlist depth must cover K plus the excluded history")
        _, retrieved = self.index.search(self.vectors[history], self.depth)
        selected = np.setdiff1d(retrieved.ravel(), history)
        scores = interest_scores(self.vectors[history] @ self.vectors[selected].T, "max_last_025")
        order = np.argsort(-scores, kind="stable")[:k]
        return seq, self.item_ids[selected[order]].tolist(), scores[order].tolist()
