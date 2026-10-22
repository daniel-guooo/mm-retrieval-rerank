"""按召回是否命中 target 拆分候选，分开测完整 pipeline 和 Qwen 本身的重排能力。

- hits：只留召回里有 target 的用户，候选原样不动。未命中用户重排后必为 0，
  所以完整 pipeline 的 HR/NDCG@k = 召回率 × hits 上的 HR/NDCG@k
- inject：只留召回里没有 target 的用户，删掉第 K 名，把 target 随机插到 1..K 的某个位置，
  其余候选保持召回顺序。随机插入位置的期望 HR@k = k/K。这是诊断用的 oracle，不是召回结果

    python scripts/make_oracle_candidates.py manifests/microlens/val_fusion_a0p5_g0p5_top50.jsonl --mode hits
    python scripts/make_oracle_candidates.py manifests/microlens/val_fusion_a0p5_g0p5_top50.jsonl --mode inject
"""
import argparse
import hashlib
from pathlib import Path
import random

from mmrr.eval.ranking import rank_of
from mmrr.io import read_jsonl, write_jsonl, write_meta


def inject(record, rng):
    candidates = list(record["candidates"][:-1])
    candidates.insert(rng.randrange(len(record["candidates"])), record["target"])
    out = {k: v for k, v in record.items() if k != "scores"}  # 插入后召回分数不再对应，重排也不用它
    return {**out, "candidates": candidates, "target_rank": rank_of(candidates, record["target"]),
            "oracle_injected": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", type=Path)
    ap.add_argument("--mode", required=True, choices=("hits", "inject"))
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    out = args.candidates.with_name(f"{args.candidates.stem}_{args.mode}.jsonl")
    if out.exists():
        ap.error(f"Output already exists: {out}")
    source = list(read_jsonl(args.candidates))
    if args.mode == "hits":
        records = [r for r in source if r["target"] in r["candidates"]]
    else:
        rng = random.Random(args.seed)
        records = [inject(r, rng) for r in source if r["target"] not in r["candidates"]]
    write_jsonl(out, records)
    write_meta(out.with_suffix(".meta.json"), {
        "mode": args.mode, "oracle": args.mode == "inject", "source_file": args.candidates.name,
        "source_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "n_users": len(records), "n_users_source": len(source),
        "seed": args.seed if args.mode == "inject" else None,
    })
    print(f"{len(records)}/{len(source)} users ({args.mode}) -> {out}")


if __name__ == "__main__":
    main()
