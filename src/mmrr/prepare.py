"""Create natural interest manifests and matched pointwise training examples."""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np

from mmrr.artifacts import new_directory, require_new, save_json, save_jsonl, sha256, verified_sidecar
from mmrr.data.loaders import load_sequences, read_interactions
from mmrr.eval.ranking import candidate_record
from mmrr.interest import retrieve_interest
from mmrr.io import read_jsonl
from mmrr.paths import DATA_ROOT
from mmrr.retrieval import item_vectors

CLIP_ARRAYS = ("item_ids", "item_text", "item_img", "item_has_img")


def clip_vectors(cache: Path, alpha: float = 0.5):
    ids, text, images, mask = (
        np.load(cache / f"{name}.npy", allow_pickle=False) for name in CLIP_ARRAYS
    )
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 2
            or np.any(ids[1:] <= ids[:-1]) or text.shape != (len(ids), 512)
            or images.shape != text.shape or mask.shape != ids.shape or mask.dtype != np.bool_):
        raise ValueError("Invalid CLIP catalog arrays")
    if not mask.all():
        raise ValueError("The interest experiment requires complete image embeddings (use MicroLens)")
    if not all(np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5) for x in (text, images)):
        raise ValueError("Expected finite, normalized CLIP embeddings")
    return ids, item_vectors(text, images, "fusion", alpha)


def export_candidates(dataset, split, cache, output, k, batch_size, max_users):
    meta = output.with_suffix(".meta.json")
    require_new(output, meta)
    ids, vectors = clip_vectors(cache)
    sequences = load_sequences(dataset, split)
    n_total = len(sequences)
    sequences = sequences[:max_users]
    if not sequences or not {i for seq in sequences for i in [*seq.history, seq.target]} <= set(ids.tolist()):
        raise ValueError("Empty split or data IDs missing from the CLIP cache")

    def records():
        for seq, items, scores in retrieve_interest(sequences, ids, vectors, k=k, batch_size=batch_size):
            if len(items) != k:
                raise ValueError(f"Fewer than {k} eligible candidates for {seq.user}")
            yield candidate_record(seq, items, scores)

    count = save_jsonl(output, records())
    save_json(meta, {
        "dataset": dataset, "split": split, "method": "max_last_025", "alpha": 0.5,
        "k": k, "n_users": count, "n_users_total": n_total, "n_items": len(ids),
        "source_sha256": {
            f"{split}.tsv": sha256(DATA_ROOT / dataset / f"{split}.tsv"),
            **{f"{name}.npy": sha256(cache / f"{name}.npy") for name in CLIP_ARRAYS},
        },
        "manifest_sha256": sha256(output),
    }, meta=True)
    return count


def matched_training_rows(candidates, observations, item_ids, *, seed, negatives, rank_low, rank_high):
    """Build matched datasets with exactly one positive and N negatives per retained user."""
    random_rng = random.Random(seed)
    retrieved_rng = random.Random(seed)
    items = sorted(set(item_ids))
    for record in candidates:
        user, target = record["user"], record["target"]
        observed = observations[user]
        if observed[-1] != target or observed[-6:-1] != record["history"]:
            raise ValueError(f"Training candidate does not match train.tsv: {user}")
        exclude = set(observed)
        ranked = [(i, rank) for rank, i in enumerate(record["candidates"], 1)
                  if rank_low <= rank <= rank_high and i not in exclude]
        if len(ranked) < negatives or len(items) - len(exclude) < negatives:
            continue
        retrieved = retrieved_rng.sample(ranked, negatives)
        random_items = []
        random_exclude = exclude.copy()
        while len(random_items) < negatives:
            item = random_rng.choice(items)
            if item not in random_exclude:
                random_items.append(item)
                random_exclude.add(item)
        natural_ranks = {item: rank for rank, item in enumerate(record["candidates"], 1)}
        common = {"user": user, "history": record["history"], "target": target}
        yield (
            {**common, "negatives": [{"item": item, "rank": natural_ranks.get(item)} for item in random_items]},
            {**common, "negatives": [{"item": item, "rank": rank} for item, rank in retrieved]},
        )


def pointwise_examples(users):
    """Produce both SFT labels and one-token DPO preferences with HaRS anchor IDs."""
    for row in users:
        target = row["target"]
        common = {"user": row["user"], "history": row["history"], "positive_item": target}
        yield {
            **common, "candidate": target, "chosen": "Yes", "rejected": "No",
            "negative_item": row["negatives"][0]["item"], "retrieval_rank": None,
        }
        for neg in row["negatives"]:
            yield {
                **common, "candidate": neg["item"], "chosen": "No", "rejected": "Yes",
                "negative_item": neg["item"], "retrieval_rank": neg["rank"],
            }


def prepare_training(dataset, candidates, cache, output_dir, seed, negatives, rank_low, rank_high):
    require_new(output_dir)
    meta = verified_sidecar(candidates, "manifest_sha256")
    if meta["dataset"] != dataset or meta["split"] != "train" or meta["method"] != "max_last_025":
        raise ValueError("Use an interest manifest generated from this dataset's training split")
    split_hash = sha256(DATA_ROOT / dataset / "train.tsv")
    id_path = cache / "item_ids.npy"
    if meta["source_sha256"]["train.tsv"] != split_hash or meta["source_sha256"]["item_ids.npy"] != sha256(id_path):
        raise ValueError("Training data or catalog IDs differ from the candidate manifest")
    if not 1 <= negatives <= rank_high - rank_low + 1 or not 1 <= rank_low <= rank_high <= meta["k"]:
        raise ValueError("Negative count and rank band must fit the manifest cutoff")
    records = list(read_jsonl(candidates))
    if len(records) != meta["n_users"] or len({row["user"] for row in records}) != len(records):
        raise ValueError("Invalid candidate users")
    ids = np.load(id_path, allow_pickle=False).tolist()
    observations = dict(read_interactions(dataset, "train"))
    pairs = list(matched_training_rows(records, observations, ids, seed=seed, negatives=negatives,
                                       rank_low=rank_low, rank_high=rank_high))
    if not pairs:
        raise ValueError("No users have enough eligible negatives in the requested rank band")
    with new_directory(output_dir) as stage:
        for index, strategy in enumerate(("random", "retrieved")):
            path = stage / f"{strategy}.jsonl"
            count = save_jsonl(path, pointwise_examples(pair[index] for pair in pairs))
            save_json(path.with_suffix(".meta.json"), {
                "dataset": dataset, "split": "train", "strategy": strategy, "seed": seed,
                "rank_band": [rank_low, rank_high], "negatives": negatives, "n_users": len(pairs),
                "n_dropped_users": len(records) - len(pairs), "n_examples": count,
                "train_sha256": split_hash, "candidate_sha256": meta["manifest_sha256"],
                "catalog_ids_sha256": meta["source_sha256"]["item_ids.npy"], "sha256": sha256(path),
            }, meta=True)
    return len(pairs)
