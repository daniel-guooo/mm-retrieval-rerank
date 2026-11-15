"""CLIP 编码全库商品，生成向量缓存。

    python scripts/encode_items.py --dataset microlens
输出 OUTPUT_ROOT/emb/clip-vit-b32/<dataset>/{item_ids,item_text,item_img,item_has_img}.npy + meta.json（行对齐）
缺图的物品 item_img 那一行全 0、item_has_img 为 False，召回时怎么处理由召回代码决定。
"""
import argparse
import time
from pathlib import Path

import numpy as np

from mmrr.data.loaders import load_catalog
from mmrr.io import write_meta
from mmrr.models.clip_encoder import DEFAULT_CLIP, ClipEncoder
from mmrr.paths import emb_dir, resolve_data_path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="microlens")
ap.add_argument("--model", default=DEFAULT_CLIP)
ap.add_argument("--revision")
ap.add_argument("--output", type=Path)
ap.add_argument("--metadata-only", action="store_true", help="Use catalog titles/images without reading split TSVs")
ap.add_argument("--text-bs", type=int, default=256)
ap.add_argument("--img-bs", type=int, default=128)
args = ap.parse_args()

if args.text_bs < 1 or args.img_bs < 1:
    ap.error("Batch sizes must be positive")
started = time.perf_counter()
catalog = load_catalog(args.dataset, include_interaction_ids=not args.metadata_only)
ids = sorted(catalog)
enc = ClipEncoder(args.model, revision=args.revision)
encoding_started = time.perf_counter()
text = enc.encode_text([catalog[i].title for i in ids], args.text_bs)
has_img = np.array([catalog[i].image_path is not None for i in ids])
img = np.zeros_like(text)
if has_img.any():
    img[has_img] = enc.encode_images(
        [resolve_data_path(catalog[i].image_path) for i, h in zip(ids, has_img) if h], args.img_bs)

encoding_seconds = time.perf_counter() - encoding_started
out = args.output or emb_dir(args.dataset)
out.mkdir(parents=True, exist_ok=True)
np.save(out / "item_ids.npy", np.asarray(ids, dtype=np.int64))
np.save(out / "item_text.npy", text)
np.save(out / "item_img.npy", img)
np.save(out / "item_has_img.npy", has_img)
write_meta(out / "meta.json", {"dataset": args.dataset, "n_items": len(ids),
                               "n_missing_images": int((~has_img).sum()), "device": enc.device,
                               "includes_interaction_ids": not args.metadata_only,
                               "encoding_seconds": encoding_seconds, "total_seconds": time.perf_counter() - started,
                               "cache_bytes": sum(p.stat().st_size for p in out.glob("item_*.npy")), **enc.meta()})
print(f"saved {len(ids)} items -> {out}  (truncated titles: {enc.n_truncated})")
