"""Portable data contracts and validation; no model dependencies."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath

from mmrr import paths


@dataclass(frozen=True)
class Item:
    item_id: int
    title: str
    image_path: str | None


@dataclass(frozen=True)
class Sequence:
    user: str
    history: list[int]
    target: int


@dataclass
class ValidationReport:
    stats: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("\n".join(self.errors))


def validate_catalog(catalog: dict[int, Item], check_files: bool = True) -> ValidationReport:
    report = ValidationReport({"items": len(catalog), "missing_images": 0})
    for key, item in catalog.items():
        if type(key) is not int or type(item.item_id) is not int or key != item.item_id:
            report.errors.append(f"Invalid item ID: {key!r}")
        if not isinstance(item.title, str) or not item.title.strip():
            report.errors.append(f"Empty title: {key}")
        rel = item.image_path
        if rel is None:
            report.stats["missing_images"] += 1
        elif not isinstance(rel, str) or not rel or "\\" in rel or PurePosixPath(rel).is_absolute() or PureWindowsPath(rel).drive or ".." in PurePosixPath(rel).parts:
            report.errors.append(f"Invalid relative image path: {key}: {rel}")
        elif check_files and not (paths.DATA_ROOT / rel).is_file():
            report.errors.append(f"Missing image file: {key}: {rel}")
    return report


def validate_sequences(sequences: list[Sequence], catalog: dict[int, Item]) -> ValidationReport:
    report = ValidationReport({"sequences": len(sequences), "history_len_dist": dict(sorted(Counter(len(s.history) for s in sequences).items())),
                               "target_in_history": sum(s.target in s.history for s in sequences)})
    for user, count in Counter(s.user for s in sequences).items():
        if count > 1:
            report.errors.append(f"用户 {user!r} 出现 {count} 次")
    for seq in sequences:
        if not isinstance(seq.user, str) or not seq.user:
            report.errors.append(f"Invalid user: {seq.user!r}")
        if not isinstance(seq.history, list) or not 1 <= len(seq.history) <= 5:
            report.errors.append(f"用户 {seq.user}: history 长度 {len(seq.history)}")
        for item_id in [*seq.history, seq.target]:
            if type(item_id) is not int or item_id not in catalog:
                report.errors.append(f"用户 {seq.user}: unknown/invalid item {item_id!r}")
    return report
