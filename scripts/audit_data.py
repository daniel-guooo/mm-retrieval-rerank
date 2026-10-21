"""Audit the immutable release and write JSON, Markdown and proposal Table 1.

python scripts/audit_data.py --tokenizer /path/to/local/clip --output /path/to/report
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from mmrr.data.loaders import (DATASETS, SPLITS, build_catalog, clean_title,
                               dataset_dir, image_files, read_interactions, read_titles)
from mmrr.data.schema import Sequence, validate_catalog, validate_sequences
from mmrr.paths import OUTPUT_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(dataset, tokenizer):
    titles, images = read_titles(dataset), image_files(dataset)
    rows = {s: read_interactions(dataset, s) for s in SPLITS}
    catalog, image_info = build_catalog(dataset, titles, images, rows)
    validate_catalog(catalog).raise_if_failed()
    used = {i for values in rows.values() for _, ids in values for i in ids}
    train_ids = {i for _, ids in rows["train"] for i in ids}
    cleaned = {i: clean_title(t) for i, t in titles.items()}
    title_counts = Counter(t for t in cleaned.values() if t)
    cleaned_changed = [i for i, t in titles.items() if cleaned[i] != t]
    all_ids = sorted(catalog)
    source_files = [dataset_dir(dataset) / f"{s}.tsv" for s in SPLITS]
    source_files += sorted(dataset_dir(dataset).glob("*_titles.csv"))
    long_titles = None
    if tokenizer is not None:
        long_titles = []
        for start in range(0, len(all_ids), 256):
            batch_ids = all_ids[start:start + 256]
            encoded = tokenizer([catalog[i].title for i in batch_ids], truncation=False, add_special_tokens=True)
            long_titles.extend(i for i, tokens in zip(batch_ids, encoded["input_ids"]) if len(tokens) > 77)
    report = {
        "dataset": dataset,
        "source_sha256": {p.name: sha256(p) for p in source_files},
        "policy": {"short_sequences": "retain length >=2; omit <2 with user IDs recorded",
                   "history": "last up to 5 before final target, unchanged source order",
                   "missing_title": "[unknown ID]", "missing_image": None,
                   "catalog": "union of titles, numeric image IDs and ALL official split IDs",
                   "interaction_order": "assumed oldest-to-newest from release convention; TSV has no timestamps, not independently verified",
                   "full_train": "read_interactions(dataset, 'train'); load_sequences is truncated"},
        "catalog": {"items": len(catalog), "title_rows": len(titles),
                    "sequence_unique_items": len(used), "image_id_count": len(images),
                    "id_min": min(all_ids, default=None), "id_max": max(all_ids, default=None),
                    "contains_zero": 0 in catalog,
                    "contiguous_from_one": bool(all_ids) and all_ids[0] == 1 and all_ids[-1] == len(all_ids),
                    "sequence_ids_without_title": sorted(used - titles.keys()),
                    "image_only_ids": sorted(images.keys() - titles.keys() - used),
                    "ids_without_title_row": sorted(catalog.keys() - titles.keys())},
        "titles": {"normalized_count": len(cleaned_changed), "normalized_ids": cleaned_changed,
                   "blank_ids": [i for i, t in cleaned.items() if not t],
                   "replacement_character_ids": [i for i, t in titles.items() if '\ufffd' in t],
                   "literal_nan_ids": [i for i, t in cleaned.items() if t.lower() == 'nan'],
                   "encoding_note": "strict UTF-8; replacement characters reported; no guessed mojibake repairs",
                   "duplicate_title_groups": sum(n > 1 for n in title_counts.values()),
                   "items_in_duplicate_groups": sum(n for n in title_counts.values() if n > 1),
                   "over_77_tokens_count": len(long_titles) if long_titles is not None else None,
                   "over_77_tokens_ids": long_titles,
                   "token_rule": "CLIP tokenizer, includes special tokens; no truncation"},
        "images": {"files_total": sum(p.is_file() for p in (dataset_dir(dataset) / "images").iterdir()),
                   "status_counts": dict(Counter(v["status"] for v in image_info.values())),
                   "mode_counts": dict(Counter(v["mode"] for v in image_info.values() if v["status"] == "ok")),
                   "tiny_definition": "either dimension <32 pixels",
                   "tiny_ids": [i for i, v in image_info.items() if v["status"] == "ok" and min(v["width"], v["height"]) < 32],
                   "bad_ids": {str(i): v for i, v in image_info.items() if v["status"] != "ok"}},
        "splits": {}, "user_overlap": {},
    }
    for split, values in rows.items():
        seqs = [Sequence(user, ids[-6:-1], ids[-1]) for user, ids in values if len(ids) >= 2]
        validation = validate_sequences(seqs, catalog)
        validation.raise_if_failed()
        report["splits"][split] = {
            "raw_sequences": len(values), "raw_users": len({u for u, _ in values}),
            "raw_interactions": sum(len(ids) for _, ids in values),
            "length_distribution": dict(sorted(Counter(len(ids) for _, ids in values).items())),
            "under_6_count": sum(len(ids) < 6 for _, ids in values),
            "retained_short_count": sum(2 <= len(ids) < 6 for _, ids in values),
            "excluded_under_2_users": [u for u, ids in values if len(ids) < 2],
            "sequences_with_consecutive_duplicates": sum(any(a == b for a, b in zip(ids, ids[1:])) for _, ids in values),
            "cold_target_count": sum(s.target not in train_ids for s in seqs) if split != "train" else None,
            **validation.stats,
        }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted({u for u, _ in rows[a]} & {u for u, _ in rows[b]})
        report["user_overlap"][f"{a}_{b}"] = {"count": len(overlap), "users": overlap}
    report["table1"] = {
        "dataset": dataset, "train": len(rows["train"]), "val": len(rows["val"]), "test": len(rows["test"]),
        "items": len(catalog), "images": report["images"]["files_total"],
        "interactions": sum(v["raw_interactions"] for v in report["splits"].values()),
        "unique_users": len({u for v in rows.values() for u, _ in v}),
    }
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    ap.add_argument("--tokenizer", help="Local CLIP tokenizer folder; no network or model weights needed")
    ap.add_argument("--output", type=Path, default=OUTPUT_ROOT / "audit")
    ap.add_argument("--archive", type=Path, help="Optional source zip to fingerprint")
    args = ap.parse_args()
    tokenizer = None
    if args.tokenizer:
        from transformers import CLIPTokenizerFast
        tokenizer = CLIPTokenizerFast.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in args.datasets:
        print(f"Auditing {dataset}...", flush=True)
        report = audit(dataset, tokenizer)
        reports.append(report)
        (args.output / f"{dataset}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["table1"]), flush=True)
    meta = {"tokenizer_checked": tokenizer is not None}
    if args.archive:
        meta["archive"] = {"name": args.archive.name, "bytes": args.archive.stat().st_size, "sha256": sha256(args.archive)}
    (args.output / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = ["# Data audit", "", "Source data was read only; IDs and official splits are unchanged.", "",
             "## Table 1 (raw release; interactions count TSV IDs)", "",
             "| Dataset | Train | Val | Test | Catalog items | Image files | Interactions | Unique users |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        t = r["table1"]
        lines.append("| " + " | ".join(str(t[k]) for k in ("dataset", "train", "val", "test", "items", "images", "interactions", "unique_users")) + " |")
    for r in reports:
        lines += ["", f"## {r['dataset']}", "", f"Image status: `{r['images']['status_counts']}`.",
                  f"Image modes: `{r['images']['mode_counts']}`; tiny images: {len(r['images']['tiny_ids'])}.",
                  f"Normalized titles: {r['titles']['normalized_count']}; blank titles: {len(r['titles']['blank_ids'])}; missing title rows: {len(r['catalog']['ids_without_title_row'])}.",
                  f"Duplicate title groups: {r['titles']['duplicate_title_groups']}; titles exceeding 77 tokens: {r['titles']['over_77_tokens_count']}.",
                  f"Replacement-character titles: {len(r['titles']['replacement_character_ids'])}; literal NaN titles: {len(r['titles']['literal_nan_ids'])} (preserved for review).",
                  f"ID range: {r['catalog']['id_min']}..{r['catalog']['id_max']}; contiguous from 1: {r['catalog']['contiguous_from_one']}; contains 0: {r['catalog']['contains_zero']}.", "",
                  "| Split | Retained | Excluded (<2) | Retained short (2-5) | Target in history | Cold targets | Consecutive duplicates |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for split, s in r["splits"].items():
            lines.append(f"| {split} | {s['sequences']} | {len(s['excluded_under_2_users'])} | {s['retained_short_count']} | {s['target_in_history']} | {s['cold_target_count']} | {s['sequences_with_consecutive_duplicates']} |")
        lines += ["", "User overlap: " + str({k: v['count'] for k, v in r['user_overlap'].items()}),
                  "", "Chronology is preserved from the release but cannot be independently verified without timestamps.",
                  "Targets already in history are retained and will be unreachable under history exclusion.",
                  "Use full read_interactions(train) for popularity; load_sequences intentionally returns only the recent window."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
