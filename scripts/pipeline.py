"""MicroLens final-project entry points. GPU dependencies are loaded only when needed."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from mmrr.artifacts import (
    atomic_path, load_json, output_lock, require_new, save_json, save_jsonl, sha256, verified_sidecar,
)
from mmrr.io import read_jsonl
from mmrr.models.qwen_prompt import TEMPLATES
from mmrr.models.qwen_scorer import DEFAULT_QWEN
from mmrr.paths import DATA_ROOT, OUTPUT_ROOT, emb_dir

QWEN_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
DATASET = "microlens"


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def positive_float(value):
    n = float(value)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return n


def nonnegative_float(value):
    n = float(value)
    if not math.isfinite(n) or n < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return n


def fraction(value):
    n = nonnegative_float(value)
    if n > 1:
        raise argparse.ArgumentTypeError("must be in [0,1]")
    return n


def preflight(args):
    from importlib.metadata import PackageNotFoundError, version

    files = [DATA_ROOT / DATASET / filename for filename in
             ("train.tsv", "val.tsv", "test.tsv", "Microlens_titles.csv", "images")]
    missing = [str(path) for path in files if not path.exists()]
    versions = {}
    for package in ("numpy", "pillow", "torch", "torchvision", "transformers",
                    "peft", "accelerate", "qwen-vl-utils"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    info = {"data_root": str(DATA_ROOT), "output_root": str(OUTPUT_ROOT),
            "dependencies": versions, "missing_data": missing}
    if args.gpu:
        import torch

        info.update({
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "visible_gpu_count": torch.cuda.device_count(),
        })
        if torch.cuda.is_available():
            info.update({"gpu": torch.cuda.get_device_name(),
                         "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                         "bf16": torch.cuda.is_bf16_supported()})
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if missing or (args.gpu and (info["visible_gpu_count"] != 1 or not all(versions.values()))):
        raise SystemExit("Preflight failed: check the data root, dependencies and visible GPU")


def candidates(args):
    from mmrr.prepare import export_candidates

    count = export_candidates(args.dataset, args.split, args.cache, args.output, args.k,
                              args.batch_size, args.max_users)
    print(f"Saved {count} candidate lists to {args.output}")


def prepare(args):
    from mmrr.prepare import prepare_training

    count = prepare_training(args.dataset, args.manifest, args.cache, args.output, args.seed,
                             args.negatives, args.rank_low, args.rank_high)
    print(f"Saved matched training examples for {count} users to {args.output}")


def encode_neighbors(args):
    from mmrr.hars import encode_hars

    encode_hars(args.dataset, args.model, args.revision, args.precision, args.output,
                args.k, args.batch_size, args.device, args.smoke_examples)


def add_hardness(args):
    from mmrr.hars import attach_hars

    attach_hars(args.examples, args.neighbors, args.output)


def train(args):
    from mmrr.train import train_adapter

    if args.max_steps == 0 or args.max_steps < -1 or args.beta <= 0:
        raise ValueError("max-steps must be -1 or positive, and beta positive")
    train_adapter(args)


def score(args):
    from mmrr.score import score_manifest

    print(score_manifest(args))


def _load_manifest(path):
    meta = verified_sidecar(path, "manifest_sha256")
    if meta["split"] not in ("val", "test"):
        raise ValueError("Evaluation requires a natural validation or test manifest")
    rows = list(read_jsonl(path))
    if len(rows) != meta["n_users"] or any(len(row["candidates"]) != meta["k"] for row in rows):
        raise ValueError("Candidate manifest size differs from its metadata")
    return rows, meta


def evaluate(args):
    from mmrr.data.loaders import read_interactions
    from mmrr.eval.experiment import evaluate_natural

    rows, meta = _load_manifest(args.manifest)
    popularity = None
    if args.popularity:
        popularity = Counter(item for _, ids in read_interactions(meta["dataset"], "train") for item in ids)
    report, per_user = evaluate_natural(rows, read_jsonl(args.scores) if args.scores else None,
                                         k=args.k, n_items=args.catalog_size or meta.get("n_items"),
                                         popularity=popularity)
    summary = args.output / "metrics.json"
    ranks = args.output / "ranks.jsonl"
    require_new(summary, ranks)
    save_jsonl(ranks, per_user)
    save_json(summary, {
        **report, "dataset": meta["dataset"], "split": meta["split"],
        "n_users_in_split": meta.get("n_users_total", meta["n_users"]),
        "candidate_sha256": meta["manifest_sha256"],
        "score_sha256": sha256(args.scores) if args.scores else None,
        "ranks_sha256": sha256(ranks),
    }, meta=True)
    print(json.dumps(report, indent=2))


def fuse(args):
    from mmrr.eval.experiment import evaluate_natural, fused_runs

    rows, meta = _load_manifest(args.manifest)
    if args.weight is None and meta["split"] != "val":
        raise ValueError("Select fusion weights on validation; set --weight explicitly on test")
    scores = list(read_jsonl(args.scores))
    options = [args.weight] if args.weight is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    comparisons = []
    for weight in options:
        ranked = fused_runs(rows, scores, weight)
        metrics, _ = evaluate_natural(rows, ranked)
        comparisons.append({"weight": weight, **metrics["ranking"]})
    selected = max(comparisons, key=lambda row: (row["NDCG@10"], -row["weight"]))["weight"]
    report = args.output.with_suffix(".meta.json")
    require_new(args.output, report)
    save_jsonl(args.output, fused_runs(rows, scores, selected))
    save_json(report, {
        "split": meta["split"], "candidate_sha256": meta["manifest_sha256"],
        "qwen_score_sha256": sha256(args.scores), "selection_metric": "NDCG@10",
        "weight": selected, "comparisons": comparisons, "sha256": sha256(args.output),
    }, meta=True)
    print(json.dumps({"selected": selected, "comparisons": comparisons}, indent=2))


def compare(args):
    from mmrr.eval.experiment import paired_bootstrap

    require_new(args.output)
    metadata = [load_json(path.with_name("metrics.json")) for path in (args.baseline, args.variant)]
    for path, meta in zip((args.baseline, args.variant), metadata, strict=True):
        if meta["ranks_sha256"] != sha256(path):
            raise ValueError(f"Rank data differs from its evaluation metadata: {path}")
    left, right = metadata
    if any(left[field] != right[field] for field in ("candidate_sha256", "split", "k")):
        raise ValueError("Paired runs must use the same natural candidates, split and cutoff")
    result = paired_bootstrap(read_jsonl(args.baseline), read_jsonl(args.variant),
                              samples=args.samples, seed=args.seed)
    save_json(args.output, {"baseline_sha256": sha256(args.baseline), "variant_sha256": sha256(args.variant),
                            "samples": args.samples, "seed": args.seed, "metrics": result}, meta=True)
    print(json.dumps(result, indent=2))


def report(args):
    """Aggregate available seeds into a table without inventing pending results."""
    from mmrr.eval.experiment import RANK_METRICS

    spec = load_json(args.matrix)
    rows = []
    common = None
    for variant in spec["variants"]:
        runs = []
        seeds = variant["seeds"]
        for seed in seeds:
            label = variant["name"] if seed is None else f"{variant['name']}_{seed}"
            path = args.results / label / "metrics.json"
            if path.exists():
                metrics = load_json(path)
                if (metrics["n_users"] != metrics["n_users_in_split"] or metrics["dataset"] != spec["dataset"]
                        or metrics["k"] != spec["k"]):
                    raise ValueError(f"{label} must use the full split and the matrix's dataset and cutoff")
                key = tuple(metrics[field] for field in ("candidate_sha256", "split", "n_users", "k"))
                if common is not None and common != key:
                    raise ValueError(f"Evaluation inputs for {label} do not match the other runs")
                common = key
                runs.append(metrics)
        present = len(runs)
        row = {"experiment": variant["name"], "runs_done": present, "runs_planned": len(seeds),
               "candidate_recall": runs[0]["candidate_recall"] if present else None}
        for name in RANK_METRICS:
            values = [r["ranking"][name] for r in runs]
            row[f"{name}_mean"] = statistics.mean(values) if present else None
            row[f"{name}_sd"] = statistics.stdev(values) if present > 1 else None
        rows.append(row)
    require_new(args.output)
    with atomic_path(args.output) as temp, temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


def benchmark(args):
    from mmrr.score import benchmark as run

    require_new(args.output)
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    save_json(args.output, run(args), meta=True)
    print(args.output)


def recommend(args):
    from mmrr.data.loaders import load_catalog
    from mmrr.data.schema import Sequence
    from mmrr.eval.ranking import candidate_record
    from mmrr.interest import FaissInterestSearch, retrieve_interest
    from mmrr.prepare import clip_vectors
    from mmrr.score import make_hf_scorer, score_batch, scorer_options, scoring_prompt

    if not 1 <= len(args.history) <= 5 or args.show > args.k:
        raise ValueError("Use one to five chronological item IDs and show <= K")
    ids, vectors = clip_vectors(args.cache)
    if not set(args.history) <= set(ids.tolist()):
        raise ValueError("Unknown history item ID")
    catalog = load_catalog(args.dataset, include_interaction_ids=False)
    seq = Sequence("demo", args.history, -1)
    if args.faiss_depth:
        search = FaissInterestSearch(ids, vectors, args.faiss_depth)
        query, items, scores = search.retrieve(seq, args.k)
    else:
        query, items, scores = next(retrieve_interest([seq], ids, vectors, k=args.k))
    if not args.clip_only:
        scorer_options(args)
        result = next(score_batch([candidate_record(query, items, scores)], catalog,
                                  make_hf_scorer(args), scoring_prompt(args), args.dataset))
        items = result["reranked"]
    print(json.dumps([{"item": item, "title": catalog[item].title} for item in items[:args.show]],
                     ensure_ascii=False, indent=2))


def add_model_arguments(parser, *, scoring=False):
    parser.add_argument("--model", default=DEFAULT_QWEN)
    parser.add_argument("--revision", default=QWEN_REVISION)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    if scoring:
        parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        parser.add_argument("--adapter", type=Path)
        parser.add_argument("--sft-adapter", type=Path)
        parser.add_argument("--batch-size", type=positive, default=1)
        parser.add_argument("--template", choices=TEMPLATES, default="onerec")
        parser.add_argument("--history-images", action="store_true")
        parser.add_argument("--no-candidate-images", action="store_true")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.set_defaults(dataset=DATASET)
    commands = root.add_subparsers(required=True, dest="command")

    def command(name, action, help_text, *, output=False):
        sub = commands.add_parser(name, help=help_text)
        sub.set_defaults(action=action)
        if output:
            sub.add_argument("--output", type=Path, required=True)
        return sub

    def cache_arg(sub):
        sub.add_argument("--cache", type=Path, default=emb_dir(DATASET))

    def manifest_arg(sub):
        sub.add_argument("--manifest", type=Path, required=True)

    sub = command("preflight", preflight, "check data paths and the optional CUDA environment")
    sub.add_argument("--gpu", action="store_true")
    sub = command("candidates", candidates, "generate a natural interest Top-K manifest", output=True)
    cache_arg(sub)
    sub.add_argument("--split", choices=("train", "val", "test"), required=True)
    sub.add_argument("--k", type=positive, default=100)
    sub.add_argument("--batch-size", type=positive, default=128)
    sub.add_argument("--max-users", type=positive)
    sub = command("prepare", prepare, "build matched random/retrieved train examples", output=True)
    cache_arg(sub)
    manifest_arg(sub)
    sub.add_argument("--seed", type=int, default=2025)
    sub.add_argument("--negatives", type=positive, default=4)
    sub.add_argument("--rank-low", type=positive, default=1)
    sub.add_argument("--rank-high", type=positive, default=50)
    sub = command("hars-neighbors", encode_neighbors, "encode the Qwen catalog and prepare neighbor distributions", output=True)
    add_model_arguments(sub)
    sub.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    sub.add_argument("--k", type=positive, default=10)
    sub.add_argument("--batch-size", type=positive, default=4)
    sub.add_argument("--smoke-examples", type=Path, help="Encode a small catalog supporting this smoke fixture")
    sub = command("hars-attach", add_hardness, "attach offline HaRS weights to preferences", output=True)
    sub.add_argument("--examples", type=Path, required=True)
    sub.add_argument("--neighbors", type=Path, required=True)
    sub = command("train", train, "train or resume a single-GPU LoRA adapter", output=True)
    add_model_arguments(sub)
    sub.add_argument("--stage", choices=("sft", "dpo", "hanorec"), required=True)
    sub.add_argument("--examples", type=Path, required=True)
    sub.add_argument("--sft-adapter", type=Path)
    sub.add_argument("--resume", action="store_true")
    sub.add_argument("--seed", type=int, default=2025)
    sub.add_argument("--batch-size", type=positive, default=2)
    sub.add_argument("--accumulation", type=positive, default=8)
    sub.add_argument("--epochs", type=positive_float, default=1)
    sub.add_argument("--max-steps", type=int, default=-1)
    sub.add_argument("--save-steps", type=positive, default=100)
    sub.add_argument("--lr", type=positive_float, default=1e-4)
    sub.add_argument("--rank", type=positive, default=8)
    sub.add_argument("--lora-alpha", type=positive, default=32)
    sub.add_argument("--lora-dropout", type=fraction, default=0.05)
    sub.add_argument("--beta", type=positive_float, default=0.1)
    sub.add_argument("--sigma", type=nonnegative_float, default=0.05)
    sub.add_argument("--max-tokens", type=positive, default=4096)
    sub = command("score", score, "score a manifest in resumable shards using HF", output=True)
    manifest_arg(sub)
    add_model_arguments(sub, scoring=True)
    sub.add_argument("--hits-only", action="store_true")
    sub.add_argument("--max-users", type=positive)
    sub.add_argument("--shard-users", type=positive, default=16)
    sub = command("evaluate", evaluate, "compute conditional and all-user ranking metrics", output=True)
    manifest_arg(sub)
    sub.add_argument("--scores", type=Path)
    sub.add_argument("--k", type=positive)
    sub.add_argument("--catalog-size", type=positive)
    sub.add_argument("--popularity", action="store_true")
    sub = command("fuse", fuse, "select a CLIP/Qwen rank-fusion weight on validation", output=True)
    manifest_arg(sub)
    sub.add_argument("--scores", type=Path, required=True)
    sub.add_argument("--weight", type=fraction)
    sub = command("compare", compare, "paired user bootstrap for two evaluated rankings", output=True)
    sub.add_argument("--baseline", type=Path, required=True)
    sub.add_argument("--variant", type=Path, required=True)
    sub.add_argument("--samples", type=positive, default=2000)
    sub.add_argument("--seed", type=int, default=2025)
    sub = command("report", report, "summarize completed experiment runs and seeds", output=True)
    sub.add_argument("--matrix", type=Path, required=True)
    sub.add_argument("--results", type=Path, required=True)
    sub = command("benchmark", benchmark, "measure warmed-up retrieval and reranking latency", output=True)
    cache_arg(sub)
    add_model_arguments(sub, scoring=True)
    sub.add_argument("--split", choices=("val", "test"), default="val")
    sub.add_argument("--k", type=positive, default=100)
    sub.add_argument("--warmup", type=int, default=5)
    sub.add_argument("--faiss-depth", type=positive, help="Optional shortlist depth per history item")
    sub.add_argument("--users", type=positive, default=30)
    sub.add_argument("--seed", type=int, default=2025)
    sub = command("recommend", recommend, "demo: retrieve and optionally rerank an item history")
    cache_arg(sub)
    add_model_arguments(sub, scoring=True)
    sub.add_argument("--history", type=positive, nargs="+", required=True)
    sub.add_argument("--k", type=positive, default=100)
    sub.add_argument("--show", type=positive, default=10)
    sub.add_argument("--faiss-depth", type=positive, help="Optional shortlist depth per history item")
    sub.add_argument("--clip-only", action="store_true")
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    if hasattr(args, "output"):
        with output_lock(args.output):
            args.action(args)
    else:
        args.action(args)
