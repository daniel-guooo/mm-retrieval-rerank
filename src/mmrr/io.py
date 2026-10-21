"""jsonl / meta 读写。文件里只存 item id，不存任何本机路径，换机器可以直接用。"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from mmrr.paths import REPO_ROOT


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def write_meta(path: Path, meta: dict) -> None:
    """每个产出文件旁边放一个 meta.json，记录怎么生成的。"""
    import importlib.metadata as md

    def ver(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    full = {
        **meta,
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "versions": {p: ver(p) for p in ("torch", "transformers", "vllm", "qwen-vl-utils")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
