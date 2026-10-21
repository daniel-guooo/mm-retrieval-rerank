"""Evaluate exact retrieval baselines; optionally export a fixed Top-50/100 manifest."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm
from mmrr.data.loaders import DATASETS, load_sequences, read_interactions
from mmrr.eval.ranking import candidate_record, rank_of
from mmrr.io import write_jsonl, write_meta
from mmrr.paths import DATA_ROOT, OUTPUT_ROOT, candidates_path, emb_dir
from mmrr.retrieval import METHODS, item_vectors, popularity_scores, retrieve, summarize


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DATASETS, default="microlens")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--method", choices=("all", *METHODS), default="all")
    ap.add_argument("--alpha", type=float, default=0.5, help="Text fusion weight")
    ap.add_argument("--gamma", type=float, default=1.0, help="Older-history discount; 1 = equal weights")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-users", type=int, help="Debug only: first N queries, full catalog retained")
    ap.add_argument("--save-candidates", action="store_true", help="Export chosen method's full-split candidates")
    ap.add_argument("--k", type=int, choices=(50, 100), default=50, help="Export size; metrics always include both cutoffs")
    ap.add_argument("--manifest-tag", default="", help="Optional unique experiment label; preserves default baseline files")
    args = ap.parse_args()
    if not 0 <= args.alpha <= 1 or not 0 < args.gamma <= 1 or args.batch_size < 1:
        ap.error("Require alpha in [0,1], gamma in (0,1], positive batch size")
    if args.max_users is not None and args.max_users < 1:
        ap.error("max-users must be positive")
    if args.save_candidates and (args.method == "all" or args.max_users is not None):
        ap.error("Save candidates only for one method on a full split")
    if args.manifest_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.manifest_tag):
        ap.error("manifest-tag accepts only letters, digits, underscores and hyphens")
    manifest = candidates_path(args.dataset, args.split, args.k)
    if args.manifest_tag:
        manifest = manifest.with_name(f"{args.split}_{args.manifest_tag}_top{args.k}.jsonl")
    if args.save_candidates and (manifest.exists() or manifest.with_suffix(".meta.json").exists()):
        ap.error(f"Frozen manifest already exists: {manifest}; will not overwrite")

    print("Loading cached vectors; no images or GPU models will be loaded.", flush=True)
    source = emb_dir(args.dataset)
    arrays = {name: np.load(source / f"{name}.npy", allow_pickle=False) for name in ("item_ids", "item_text", "item_img", "item_has_img")}
    ids, text, image, mask = (arrays[n] for n in ("item_ids", "item_text", "item_img", "item_has_img"))
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 101:
        raise ValueError("Expected integer catalog IDs and >=101 items")
    if mask.shape != ids.shape or mask.dtype != np.bool_:
        raise ValueError("Invalid image availability mask")
    if text.shape != (len(ids), 512) or image.shape != text.shape:
        raise ValueError("Embedding shapes do not match catalog")
    for a in (text, image):
        if not np.isfinite(a).all():
            raise ValueError("Non-finite embeddings")
    if not np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-5):
        raise ValueError("Text embeddings must be normalized")
    if not np.allclose(np.linalg.norm(image[mask], axis=1), 1, atol=1e-5) or np.any(image[~mask] != 0):
        raise ValueError("Image embeddings/mask inconsistent")
    # Do not silently invent a missing-image fusion policy for secondary datasets.
    if not mask.all() and args.method in ("all", "image", "fusion"):
        raise ValueError("Missing images: define and document a retrieval fallback policy first")
    sequences = load_sequences(args.dataset, args.split)
    n_all = len(sequences)
    if args.max_users:
        sequences = sequences[:args.max_users]
    catalog_ids = set(ids.tolist())
    for seq in sequences:
        if any(i not in catalog_ids for i in [*seq.history, seq.target]):
            raise ValueError(f"Query {seq.user} contains an ID missing from the embedding cache")
    methods = METHODS if args.method == "all" else (args.method,)
    pop = popularity_scores(ids, read_interactions(args.dataset, "train")) if "popularity" in methods else None
    tag = f"{args.method}_a{args.alpha:g}_g{args.gamma:g}"
    if args.max_users:
        tag += f"_first{args.max_users}"
    output = OUTPUT_ROOT / "retrieval" / args.dataset / args.split / tag
    output.mkdir(parents=True, exist_ok=True)
    print(f"Catalog={len(ids)}, queries={len(sequences)}/{n_all}, methods={','.join(methods)}", flush=True)
    config = {"dataset": args.dataset, "split": args.split, "alpha": args.alpha, "gamma": args.gamma,
              "n_users": len(sequences), "n_users_total": n_all, "n_items": len(ids),
              "max_users": args.max_users, "sampling": "first N source rows" if args.max_users else "all queries",
              "history_exclusion": "input history only", "tie_break": "ascending original item ID",
              "search": "exact CPU inner product", "batch_size": args.batch_size,
              "metric_scope": "all queries; unretrieved targets score zero",
              "popularity_source": "full train.tsv interactions only",
              "embedding_meta": json.loads((source / "meta.json").read_text(encoding="utf-8")),
              "source_sha256": {p.name: digest(p) for p in [*sorted(source.glob("*.npy")), DATA_ROOT / args.dataset / f"{args.split}.tsv"]},
              "retrieval_code_sha256": digest(Path(__file__).resolve().parents[1] / "src/mmrr/retrieval.py")}
    if "popularity" in methods:
        config["source_sha256"]["train.tsv"] = digest(DATA_ROOT / args.dataset / "train.tsv")
    summary = []
    saved_records = []
    for method in methods:
        vectors = None if method == "popularity" else item_vectors(text, image, method, args.alpha)
        ranks, covered = [], set()
        start = time.perf_counter()
        records = retrieve(sequences, ids, vectors=vectors, popularity=pop if method == "popularity" else None,
                           gamma=args.gamma, k=100, batch_size=args.batch_size)
        for seq, candidates, scores in tqdm(records, total=len(sequences), desc=method):
            ranks.append(rank_of(candidates, seq.target))
            covered.update(candidates[:50])
            if args.save_candidates:
                saved_records.append(candidate_record(seq, candidates[:args.k], scores[:args.k]))
        seconds = time.perf_counter() - start
        row = {"method": method, **summarize(ranks, covered, len(ids)), "seconds": seconds,
               "ms_per_user": seconds * 1000 / len(sequences)}
        summary.append(row)
        print(json.dumps(row), flush=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_meta(output / "meta.json", config)
    if args.save_candidates:
        write_jsonl(manifest, saved_records)
        write_meta(manifest.with_suffix(".meta.json"), {**config, "method": args.method, "k": args.k,
                                                       "manifest_tag": args.manifest_tag,
                                                       "manifest_sha256": digest(manifest)})
        print(f"Saved fixed Top-{args.k}: {manifest}", flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
