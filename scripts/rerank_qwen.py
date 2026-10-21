"""Qwen2.5-VL-3B zero-shot 重排召回候选。

    # 先抽 1000 个用户测速
    python scripts/rerank_qwen.py --dataset microlens --split test --max-users 1000
    # 检查前缀缓存没有算错（关掉缓存跑同样 20 个用户，比分数）
    python scripts/rerank_qwen.py --split val --max-users 20 --tag check_cache_on
    python scripts/rerank_qwen.py --split val --max-users 20 --tag check_cache_off --no-prefix-cache

输入 manifests/<dataset>/<split>_top50.jsonl
输出 OUTPUT_ROOT/rerank/<tag>/<dataset>/<split>.jsonl + <split>.meta.json
"""
import argparse
import hashlib
from pathlib import Path
import random
import time

from tqdm import tqdm

from mmrr.data.loaders import load_catalog
from mmrr.data.prompt_dataset import RerankPromptDataset
from mmrr.eval.ranking import rerank_record
from mmrr.io import read_jsonl, write_jsonl, write_meta
from mmrr.models.qwen_prompt import TEMPLATES, PromptConfig
from mmrr.models.qwen_scorer import DEFAULT_QWEN, make_scorer
from mmrr.paths import candidates_path, rerank_dir

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="microlens")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--candidates", type=Path, help="Explicit candidate manifest, including tagged experiments")
ap.add_argument("--hf-batch-size", type=int, default=1, help="Candidate prompts per HF forward pass")
ap.add_argument("--template", default="onerec", choices=list(TEMPLATES))
ap.add_argument("--history-images", action="store_true")
ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
ap.add_argument("--model", default=DEFAULT_QWEN)
ap.add_argument("--max-users", type=int, default=None, help="随机抽多少个用户；不填就全量")
ap.add_argument("--seed", type=int, default=2025)
ap.add_argument("--users-per-call", type=int, default=32, help="每次送进模型的用户数")
ap.add_argument("--no-prefix-cache", action="store_true")
ap.add_argument("--tag", default=None, help="输出目录名，默认 qwen25vl3b_zs_<template>")
args = ap.parse_args()
if args.users_per_call < 1 or args.hf_batch_size < 1 or (args.max_users is not None and args.max_users < 1):
    ap.error("Batch sizes and max-users must be positive")
out = rerank_dir(args.dataset, args.tag or f"qwen25vl3b_zs_{args.template}")
if (out / f"{args.split}.jsonl").exists() or (out / f"{args.split}.meta.json").exists():
    ap.error(f"Output already exists; choose a new --tag: {out}")

candidate_file = args.candidates or candidates_path(args.dataset, args.split, args.k)
cands = list(read_jsonl(candidate_file))
if not cands or any(len(r['candidates']) != args.k for r in cands):
    ap.error("Empty manifest or candidate count does not match --k")
n_all = len(cands)
if args.max_users and args.max_users < n_all:
    keep = set(random.Random(args.seed).sample(range(n_all), args.max_users))
    cands = [c for i, c in enumerate(cands) if i in keep]  # 保持原文件顺序

cfg = PromptConfig(template=args.template, history_images=args.history_images)
ds = RerankPromptDataset(cands, load_catalog(args.dataset), cfg, args.dataset)

kw = {"model_name": args.model}
if args.backend == "hf":
    kw["batch_size"] = args.hf_batch_size
if args.backend == "vllm":
    kw.update(max_images_per_prompt=ds.max_images_per_prompt, min_pixels=cfg.min_pixels,
              max_pixels=cfg.max_pixels, enable_prefix_caching=not args.no_prefix_cache)
scorer = make_scorer(args.backend, **kw)

records, t0 = [], time.perf_counter()
n_batches = -(-ds.n_users // args.users_per_call)
for recs, exs in tqdm(ds.iter_user_batches(args.users_per_call), total=n_batches, desc="rerank"):
    scores = scorer.score_messages([e.messages for e in exs])
    i = 0
    for r in recs:
        n = len(r["candidates"])
        records.append(rerank_record(r, scores[i:i + n]))
        i += n
elapsed = time.perf_counter() - t0

out = rerank_dir(args.dataset, args.tag or f"qwen25vl3b_zs_{args.template}")
write_jsonl(out / f"{args.split}.jsonl", records)
write_meta(out / f"{args.split}.meta.json", {
    "candidate_file": candidate_file.name,
    "candidate_sha256": hashlib.sha256(candidate_file.read_bytes()).hexdigest(),
    **ds.meta(), "split": args.split, "k": args.k,
    "n_users_total": n_all, "sample_seed": args.seed if args.max_users else None,
    "seconds_total": round(elapsed, 2),
    "seconds_per_user": round(elapsed / max(1, ds.n_users), 4),
    "qwen_calls_per_user": len(ds) / max(1, ds.n_users),
    **scorer.meta(),
})
print(f"{ds.n_users} users, {elapsed:.1f}s ({elapsed / max(1, ds.n_users):.3f}s/user) -> {out}")
