"""I/O for portable experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from mmrr.io import write_jsonl, write_meta


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


@contextmanager
def atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def new_directory(path: Path):
    """Publish a complete new stage directory with a single rename."""
    require_new(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=f".{path.name}.") as tmp:
        stage = Path(tmp) / path.name
        stage.mkdir()
        yield stage
        stage.rename(path)


def save_json(path: Path, data, *, meta: bool = False) -> None:
    with atomic_path(path) as temp:
        if meta:
            write_meta(temp, data)
        else:
            temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows) -> int:
    with atomic_path(path) as temp:
        return write_jsonl(temp, rows)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_new(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")


def verified_sidecar(path: Path, hash_key: str = "sha256") -> dict:
    meta = load_json(path.with_suffix(".meta.json"))
    if meta[hash_key] != sha256(path):
        raise ValueError(f"Checksum mismatch: {path}")
    return meta


def adapter_fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    hashes = [sha256(path / name) for name in ("adapter_config.json", "adapter_model.safetensors")]
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


@contextmanager
def output_lock(path: Path):
    """Exclude simultaneous writes and release the lock if the process exits."""
    lock = path.with_name(f".{path.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
