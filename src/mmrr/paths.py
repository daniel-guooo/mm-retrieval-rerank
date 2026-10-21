"""所有路径都从这里取，代码里不写死绝对路径。

默认值相对于 repo 根目录；每台机器可以用环境变量覆盖：
    MMRR_DATA_ROOT    数据包解压的位置（默认 <repo>/data）
    MMRR_OUTPUT_ROOT  向量缓存、重排结果等大文件（默认 <repo>/outputs）
manifests/ 固定在 repo 里，进 git，所有人共用。
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MMRR_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()
OUTPUT_ROOT = Path(os.environ.get("MMRR_OUTPUT_ROOT", REPO_ROOT / "outputs")).expanduser().resolve()
MANIFEST_DIR = REPO_ROOT / "manifests"


def resolve_data_path(rel: str | Path) -> Path:
    """把 catalog 里存的相对路径（相对 DATA_ROOT）变成本机的绝对路径。"""
    p = Path(rel)
    return p if p.is_absolute() else DATA_ROOT / p


def candidates_path(dataset: str, split: str, k: int = 50) -> Path:
    return MANIFEST_DIR / dataset / f"{split}_top{k}.jsonl"


def emb_dir(dataset: str, model_tag: str = "clip-vit-b32") -> Path:
    return OUTPUT_ROOT / "emb" / model_tag / dataset


def rerank_dir(dataset: str, run_tag: str = "qwen25vl3b_zeroshot") -> Path:
    return OUTPUT_ROOT / "rerank" / run_tag / dataset
