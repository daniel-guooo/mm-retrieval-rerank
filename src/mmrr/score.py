"""Resumable HF ranking for frozen natural candidate manifests."""
from __future__ import annotations

import time
from pathlib import Path

from mmrr.artifacts import adapter_fingerprint, load_json, save_json, save_jsonl, sha256
from mmrr.data.loaders import load_catalog
from mmrr.data.prompt_dataset import RerankPromptDataset
from mmrr.eval.experiment import natural_rows
from mmrr.eval.ranking import rerank_record
from mmrr.io import git_commit, read_jsonl
from mmrr.models.qwen_prompt import PromptConfig
from mmrr.models.qwen_scorer import QwenScorerHF


def scoring_prompt(args):
    return PromptConfig(template=args.template, history_images=args.history_images,
                        candidate_image=not args.no_candidate_images)


def scorer_options(args):
    main_adapter = adapter_fingerprint(args.adapter)
    sft_adapter = adapter_fingerprint(args.sft_adapter)
    for path, expected in ((args.sft_adapter, None), (args.adapter, sft_adapter)):
        if path is None or not (path / "mmrr.json").is_file():
            continue
        metadata = load_json(path / "mmrr.json")
        if metadata["sft_adapter_sha256"] != expected:
            raise ValueError("The scored adapter requires its original SFT parent")
        if metadata["revision"] != args.revision:
            raise ValueError("Scoring revision differs from the training revision")
    return {"model": args.model, "revision": args.revision, "precision": args.precision,
            "adapter_sha256": main_adapter, "sft_adapter_sha256": sft_adapter,
            "batch_size": args.batch_size, "device": args.device}


def make_hf_scorer(args):
    return QwenScorerHF(model_name=args.model, revision=args.revision, precision=args.precision,
                        batch_size=args.batch_size, adapter=str(args.adapter) if args.adapter else None,
                        sft_adapter=str(args.sft_adapter) if args.sft_adapter else None, device=args.device)


def score_batch(records, catalog, scorer, prompt, dataset):
    pairs = RerankPromptDataset(records, catalog, prompt, dataset)
    for group, examples in pairs.iter_user_batches(1):
        scores = scorer.score_messages([example.messages for example in examples])
        yield rerank_record(group[0], scores)


def _check_shard(result, records):
    if len(result) != len(records):
        raise ValueError("Incomplete scoring shard")
    for row, record in zip(result, records, strict=True):
        if (row["user"] != record["user"] or row["target"] != record["target"]
                or row["candidates"] != record["candidates"]):
            raise ValueError("Scoring shard candidate mismatch")


def score_manifest(args, scorer_factory=make_hf_scorer, catalog_loader=load_catalog):
    manifest_meta = load_json(args.manifest.with_suffix(".meta.json"))
    candidate_hash = sha256(args.manifest)
    if manifest_meta["manifest_sha256"] != candidate_hash or manifest_meta["dataset"] != args.dataset:
        raise ValueError("Scoring requires the original natural manifest and its sidecar")
    if manifest_meta["split"] not in ("val", "test"):
        raise ValueError("Score natural validation and test candidates")
    full = list(natural_rows(read_jsonl(args.manifest)).values())
    if len(full) != manifest_meta["n_users"] or any(len(r["candidates"]) != manifest_meta["k"] for r in full):
        raise ValueError("Candidate manifest size differs from its metadata")
    records = [row for row in full if not args.hits_only or row["target_rank"] is not None]
    records = records[:args.max_users]
    prompt = scoring_prompt(args)
    options = {
        **scorer_options(args), "prompt": prompt.to_dict(), "dataset": args.dataset,
        "candidate_sha256": candidate_hash, "hits_only": args.hits_only,
        "split": manifest_meta["split"], "n_users_in_manifest": len(full), "n_users_to_score": len(records),
        "max_users": args.max_users, "shard_users": args.shard_users, "code_commit": git_commit(),
        "scorer_code_sha256": sha256(Path(__file__).resolve().parent / "models" / "qwen_scorer.py"),
        "prompt_code_sha256": sha256(Path(__file__).resolve().parent / "models" / "qwen_prompt.py"),
    }
    output = args.output
    config = output / "run.json"
    if config.exists() and load_json(config)["config"] != options:
        raise ValueError("Existing scoring run has different inputs/settings; choose another output directory")
    if not config.exists():
        save_json(config, {"config": options}, meta=True)
    scorer = None
    scores_path = output / "scores.jsonl"
    for start in range(0, len(records), args.shard_users):
        chunk = records[start:start + args.shard_users]
        shard = output / "shards" / f"{start:08d}.jsonl"
        if shard.exists():
            _check_shard(list(read_jsonl(shard)), chunk)
            continue
        if scorer is None:
            catalog = catalog_loader(args.dataset, include_interaction_ids=False)
            scorer = scorer_factory(args)
        save_jsonl(shard, score_batch(chunk, catalog, scorer, prompt, args.dataset))
        print(f"Scored {start + len(chunk)}/{len(records)} users", flush=True)
    if not scores_path.exists():
        save_jsonl(scores_path, (row for path in sorted((output / "shards").glob("*.jsonl")) for row in read_jsonl(path)))
    _check_shard(list(read_jsonl(scores_path)), records)
    save_json(scores_path.with_suffix(".meta.json"), {
        **options, "sha256": sha256(scores_path), "n_users": len(records), "complete": True,
    }, meta=True)
    return scores_path


def latency_summary(seconds):
    import numpy as np

    if not seconds:
        raise ValueError("Benchmark requires at least one timed query")
    p50, p95 = np.percentile(seconds, [50, 95])
    return {"total_seconds": sum(seconds), "mean_ms": 1000 * float(np.mean(seconds)),
            "p50_ms": 1000 * float(p50), "p95_ms": 1000 * float(p95)}


def benchmark(args):
    import random

    import torch

    from mmrr.data.loaders import load_sequences
    from mmrr.eval.ranking import candidate_record
    from mmrr.interest import FaissInterestSearch, retrieve_interest
    from mmrr.prepare import clip_vectors

    started = time.perf_counter()
    options = scorer_options(args)
    ids, vectors = clip_vectors(args.cache)
    lookup = {int(item): row for row, item in enumerate(ids)}
    faiss = FaissInterestSearch(ids, vectors, args.faiss_depth) if args.faiss_depth else None
    sequences = load_sequences(args.dataset, args.split)
    count = args.users + args.warmup
    if count > len(sequences):
        raise ValueError("Benchmark sample exceeds split size")
    sample = random.Random(args.seed).sample(sequences, count)
    prompt = scoring_prompt(args)
    catalog = load_catalog(args.dataset, include_interaction_ids=False)
    scorer = make_hf_scorer(args)
    setup_seconds = time.perf_counter() - started
    durations = []
    returned = []
    for i, seq in enumerate(sample):
        if i == args.warmup and args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        if faiss is not None:
            q, items, scores = faiss.retrieve(seq, k=args.k)
        else:
            q, items, scores = next(retrieve_interest([seq], ids, vectors, k=args.k, lookup=lookup))
        record = candidate_record(q, items, scores)
        retrieved = time.perf_counter()
        list(score_batch([record], catalog, scorer, prompt, args.dataset))
        done = time.perf_counter()
        if i >= args.warmup:
            durations.append((retrieved - start, done - retrieved, done - start))
            returned.append(items)
    result = {
        **options, "prompt": prompt.to_dict(), "n_users": args.users, "warmup": args.warmup,
        "setup_seconds": setup_seconds, "k": args.k, "seed": args.seed, "split": args.split,
        "retrieval_engine": "faiss_shortlist" if faiss is not None else "exact",
        "faiss_depth": faiss.depth if faiss is not None else None,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "retrieval": latency_summary([row[0] for row in durations]),
        "rerank": latency_summary([row[1] for row in durations]),
        "end_to_end": latency_summary([row[2] for row in durations]),
        "queries_per_second": args.users / sum(row[2] for row in durations),
        "candidates_scored": args.users * args.k,
    }
    if faiss is not None:
        timed = sample[args.warmup:]
        exact = list(retrieve_interest(timed, ids, vectors, k=args.k, lookup=lookup))
        result["exact_comparison"] = {
            "candidate_overlap": sum(len(set(items).intersection(exact_items)) for items, (_, exact_items, _) in
                                     zip(returned, exact, strict=True)) / (args.users * args.k),
            "faiss_target_hits": sum(seq.target in items for seq, items in zip(timed, returned, strict=True)),
            "exact_target_hits": sum(seq.target in items for seq, items, _ in exact),
        }
    return result
