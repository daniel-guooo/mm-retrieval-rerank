"""给数据负责人用：实现完 load_catalog / load_sequences 后跑这个，全部 OK 就算交付。

    python scripts/check_loader.py --dataset microlens
"""
import argparse
import json

from mmrr.data.loaders import SPLITS, load_catalog, load_sequences
from mmrr.data.schema import validate_catalog, validate_sequences
from mmrr.paths import DATA_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="microlens")
ap.add_argument("--skip-files", action="store_true", help="不检查图片文件是否存在")
args = ap.parse_args()

print(f"DATA_ROOT = {DATA_ROOT}")
catalog = load_catalog(args.dataset)
rep = validate_catalog(catalog, check_files=not args.skip_files)
print("catalog:", json.dumps(rep.stats, ensure_ascii=False))
rep.raise_if_failed()

for split in SPLITS:
    seqs = load_sequences(args.dataset, split)
    rep = validate_sequences(seqs, catalog)
    print(f"{split}:", json.dumps(rep.stats, ensure_ascii=False))
    rep.raise_if_failed()
    again = load_sequences(args.dataset, split)
    assert again == seqs, f"{split}: 两次调用结果不一致"
print("OK")
