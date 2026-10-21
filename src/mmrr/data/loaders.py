"""Deterministic readers for the released HaNoRec datasets.

No IDs are remapped; source rows and interactions are never reordered.
Short sequences with >=2 events are retained. Missing titles use [unknown ID].
The catalog is the union of title, image and all official split IDs.
"""
from __future__ import annotations

import csv
import html
import unicodedata
from pathlib import Path

from PIL import Image

from mmrr import paths
from mmrr.data.schema import Item, Sequence

DATASETS = ("microlens", "movielens", "netflix")
SPLITS = ("train", "val", "test")


def dataset_dir(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose {DATASETS}")
    directory = paths.DATA_ROOT / dataset
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing {directory}. MMRR_DATA_ROOT must contain dataset directories.")
    return directory


def read_interactions(dataset: str, split: str) -> list[tuple[str, list[int]]]:
    """Full source interactions, for train popularity and auditing (no truncation)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose {SPLITS}")
    source = dataset_dir(dataset) / f"{split}.tsv"
    result, seen = [], set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(row) != 2 or not row[0].strip():
                raise ValueError(f"{source}:{line}: expected user and sequence columns")
            user = row[0].strip()
            if user in seen:
                raise ValueError(f"{source}:{line}: duplicate user {user!r}")
            seen.add(user)
            try:
                ids = [int(value) for value in row[1].split()]
            except ValueError as exc:
                raise ValueError(f"{source}:{line}: non-integer item ID") from exc
            result.append((user, ids))
    return result


def clean_title(raw: str | None) -> str:
    # Preserve literal 'NaN' titles: CSV strings are not coerced to floating point.
    return " ".join(unicodedata.normalize("NFC", html.unescape(raw or "")).split())


def read_titles(dataset: str) -> dict[int, str]:
    matches = sorted(dataset_dir(dataset).glob("*_titles.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one *_titles.csv in {dataset_dir(dataset)}, found {len(matches)}")
    result = {}
    with matches[0].open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"item", "title"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{matches[0]} must have item,title columns")
        for line, row in enumerate(reader, 2):
            if row.get("item") is None or row.get("title") is None or None in row:
                raise ValueError(f"{matches[0]}:{line}: malformed CSV row")
            item_id = int(row["item"])
            if item_id in result:
                raise ValueError(f"{matches[0]}:{line}: duplicate item ID {item_id}")
            result[item_id] = row["title"]
    return result


def image_files(dataset: str) -> dict[int, Path]:
    result = {}
    for path in sorted((dataset_dir(dataset) / "images").iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            item_id = int(path.stem)
        except ValueError:
            continue
        if item_id in result:
            raise ValueError(f"Multiple image files for item {item_id}: {result[item_id]}, {path}")
        result[item_id] = path
    return result


def inspect_image(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"status": "missing"}
    if path.stat().st_size == 0:
        return {"status": "zero_bytes"}
    try:
        with Image.open(path) as image:
            image.load()  # Full decode catches corrupt pixel data, not just bad headers.
            return {"status": "ok", "mode": image.mode, "width": image.width, "height": image.height}
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return {"status": "unreadable", "error": type(exc).__name__}


def build_catalog(dataset: str, titles: dict, images: dict, rows: dict) -> tuple[dict[int, Item], dict]:
    all_ids = set(titles) | set(images)
    for split_rows in rows.values():
        all_ids.update(i for _, ids in split_rows for i in ids)
    catalog, image_info = {}, {}
    for item_id in sorted(all_ids):
        info = inspect_image(images.get(item_id))
        image_info[item_id] = info
        rel = images[item_id].relative_to(paths.DATA_ROOT).as_posix() if info["status"] == "ok" else None
        catalog[item_id] = Item(item_id, clean_title(titles.get(item_id)) or f"[unknown {item_id}]", rel)
    return catalog, image_info


def load_catalog(dataset: str) -> dict[int, Item]:
    return build_catalog(dataset, read_titles(dataset), image_files(dataset),
                         {split: read_interactions(dataset, split) for split in SPLITS})[0]


def load_sequences(dataset: str, split: str) -> list[Sequence]:
    return [Sequence(user, ids[-6:-1], ids[-1])
            for user, ids in read_interactions(dataset, split) if len(ids) >= 2]
