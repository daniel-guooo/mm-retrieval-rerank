"""HaRS preprocessing adapted from the companion HaNoRec reference implementation."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from mmrr.artifacts import atomic_path, require_new, save_json, save_jsonl, sha256, verified_sidecar
from mmrr.data.loaders import load_catalog
from mmrr.io import read_jsonl
from mmrr.models.qwen_scorer import load_qwen
from mmrr.paths import resolve_data_path


def _normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1)


def neighbor_probabilities(vectors, k, block_size=128):
    """Sorted cosine Top-K score distributions, excluding each anchor itself."""
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or not 0 < k < len(vectors) or block_size < 1:
        raise ValueError("Require finite catalog vectors, positive batch size, and 0 < K < catalog size")
    vectors = _normalize(vectors)
    probabilities = np.empty((len(vectors), k), dtype=np.float32)
    for start in range(0, len(vectors), block_size):
        stop = min(start + block_size, len(vectors))
        scores = vectors[start:stop] @ vectors.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        best = np.partition(scores, -k, axis=1)[:, -k:]
        best.sort(axis=1)
        weights = np.exp(best - best[:, -1:])
        probabilities[start:stop] = weights / weights.sum(axis=1, keepdims=True)
    return probabilities


def preference_hardness(rows, item_ids, probabilities):
    lookup = {int(item): row for row, item in enumerate(item_ids)}
    if not rows or len(lookup) != len(item_ids) or len(item_ids) != len(probabilities):
        raise ValueError("Require preference rows and unique IDs aligned with neighbor probabilities")
    positive = [lookup[row["positive_item"]] for row in rows]
    negative = [lookup[row["negative_item"]] for row in rows]
    distance = np.linalg.norm(probabilities[positive] - probabilities[negative], axis=1).astype(np.float64)
    return (1 + np.exp(-distance.mean())) / (1 + np.exp(-distance))


def encode_hars(dataset, model_name, revision, precision, output: Path, k, batch_size,
                device="cuda", smoke_examples=None):
    import torch
    from PIL import Image
    from tqdm import tqdm

    require_new(output, output.with_suffix(".meta.json"))
    if batch_size < 1:
        raise ValueError("HaRS embedding batch size must be positive")
    catalog = load_catalog(dataset, include_interaction_ids=False)
    ids = sorted(catalog)
    if smoke_examples is not None:
        meta = verified_sidecar(smoke_examples)
        if meta["dataset"] != dataset or meta["n_users"] > 32 or meta["split"] != "train":
            raise ValueError("Smoke HaRS encoding expects at most 32 prepared training users")
        anchors = {row[field] for row in read_jsonl(smoke_examples) for field in ("positive_item", "negative_item")}
        ids = sorted(anchors.union(ids[:max(32, k + 1)]))
    if not 0 < k < len(ids):
        raise ValueError("HaRS neighbor K must be smaller than the catalog")
    started = time.perf_counter()
    model, processor = load_qwen(model_name, device, precision, revision)
    model.eval()
    vectors = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(ids), batch_size), desc="HaRS catalog"):
            items = [catalog[item] for item in ids[start:start + batch_size]]
            tokens = processor.tokenizer([item.title for item in items], padding=True,
                                         truncation=True, max_length=4096, return_tensors="pt").to(device)
            states = model.get_input_embeddings()(tokens["input_ids"]).float()
            mask = tokens["attention_mask"].unsqueeze(-1)
            text = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
            visual = torch.zeros_like(text)
            present = [i for i, item in enumerate(items) if item.image_path is not None]
            if present:
                images = []
                for i in present:
                    with Image.open(resolve_data_path(items[i].image_path)) as image:
                        images.append(image.convert("RGB"))
                inputs = processor.image_processor(images=images, return_tensors="pt").to(device)
                grids = inputs["image_grid_thw"]
                image_states = model.visual(inputs["pixel_values"].to(model.visual.dtype), grids)
                merge = processor.image_processor.merge_size
                lengths = (grids.prod(1) // (merge ** 2)).tolist()
                visual[present] = torch.stack([tokens.float().mean(0) for tokens in image_states.split(lengths)])
            vectors.append(_normalize(_normalize(text.cpu().numpy()) + _normalize(visual.cpu().numpy())))
    del model
    probabilities = neighbor_probabilities(np.concatenate(vectors), k)
    with atomic_path(output) as tmp, tmp.open("wb") as stream:
        np.savez_compressed(stream, item_ids=np.asarray(ids, dtype=np.int64), probabilities=probabilities)
    save_json(output.with_suffix(".meta.json"), {
        "dataset": dataset, "model": model_name, "revision": revision, "precision": precision,
        "k": k, "n_items": len(ids), "seconds": time.perf_counter() - started, "sha256": sha256(output),
        "scope": "smoke" if smoke_examples is not None else "catalog",
        "smoke_preference_sha256": sha256(smoke_examples) if smoke_examples is not None else None,
    }, meta=True)


def attach_hars(examples: Path, neighbors: Path, output: Path):
    require_new(output, output.with_suffix(".meta.json"))
    dataset_meta = verified_sidecar(examples)
    neighbor_meta = verified_sidecar(neighbors)
    if dataset_meta["split"] != "train" or dataset_meta["dataset"] != neighbor_meta["dataset"]:
        raise ValueError("HaRS expects training preferences and matching catalog neighbors")
    if neighbor_meta["scope"] == "smoke" and neighbor_meta["smoke_preference_sha256"] != dataset_meta["sha256"]:
        raise ValueError("Smoke neighbors are specific to the training examples used to encode them")
    rows = list(read_jsonl(examples))
    with np.load(neighbors, allow_pickle=False) as data:
        weights = preference_hardness(rows, data["item_ids"], data["probabilities"])
    save_jsonl(output, ({**row, "hardness": float(weight)} for row, weight in zip(rows, weights, strict=True)))
    save_json(output.with_suffix(".meta.json"), {
        **dataset_meta, "preference_sha256": dataset_meta["sha256"],
        "neighbors_sha256": neighbor_meta["sha256"], "neighbors_model": neighbor_meta["model"],
        "neighbors_revision": neighbor_meta["revision"],
        "sha256": sha256(output), "hars_k": neighbor_meta["k"], "hars_scope": neighbor_meta["scope"],
    }, meta=True)
