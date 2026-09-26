"""Durable append-only tensor shards with a committed JSONL index.

Callers own manifests and tensor schemas. A byte watermark makes an append
visible only after its index and payload are durable. Existing JSONL stores
without a watermark remain readable; a writer establishes it before appending.
Readers never repair files. Exclusive writers discard only uncommitted tails.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    atomic_bytes(
        path,
        (
            json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )


def atomic_torch_save(path: Path, value: Mapping[str, Tensor]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(dict(value), stream)
            stream.flush()
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _marker(path: Path) -> Path:
    return path.with_name(path.name + ".commit")


def _size(path: Path) -> int | None:
    try:
        value = json.loads(_marker(path).read_bytes())
    except FileNotFoundError:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid committed index byte count: {path}.")
    return value


def index_bytes(path: Path) -> bytes:
    size = _size(path)
    try:
        with path.open("rb") as stream:
            payload = stream.read() if size is None else stream.read(size)
    except FileNotFoundError:
        if size not in (None, 0):
            raise ValueError(f"committed index is missing: {path}.") from None
        return b""
    if size is not None and len(payload) != size:
        raise ValueError(f"index is shorter than its committed byte count: {path}.")
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"append-only index ends with a partial record: {path}.")
    return payload


def read_index(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    for number, line in enumerate(index_bytes(path).splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid index line {number}: {path}.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"index line {number} must be an object: {path}.")
        rows.append(value)
    return tuple(rows)


def append_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    """Append under the caller's writer lock; publish after fsync."""
    size = _size(path)
    if size is None:
        size = len(index_bytes(path))
        atomic_bytes(_marker(path), str(size).encode())
    payload = b"".join(
        (
            json.dumps(
                value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            + "\n"
        ).encode()
        for value in values
    )
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) < size:
            raise ValueError(f"index is shorter than its committed byte count: {path}.")
        stream.truncate(size)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_bytes(_marker(path), str(size + len(payload)).encode())


class ShardWriter:
    """Exclusive writer lifecycle and contiguous sample-index commit protocol.

    Subclasses validate domain records and tensor payloads before ``commit``.
    ``truncate`` is reserved for recovery under an enclosing group commit;
    immutable payload shards remain available to existing reader snapshots.
    """

    def __init__(self, root: str | Path, *, blocking: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "shards").mkdir(exist_ok=True)
        self._lock = (self.root / ".writer.lock").open("a+b")
        try:
            fcntl.flock(
                self._lock.fileno(),
                fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
            )
            self._rows = list(read_index(self.root / "index.jsonl"))
            self._reset()
            numbers = []
            for path in (self.root / "shards").glob("*.pt"):
                if path.name.startswith("."):
                    continue
                if not path.stem.isdigit():
                    raise ValueError(f"unexpected shard name: {path.name}.")
                numbers.append(int(path.stem))
            self._next_shard = max(numbers, default=-1) + 1
            self._failed = False
        except BaseException:
            self.close()
            raise

    def _reset(self) -> None:
        self._sample_ids: set[str] = set()
        for index, row in enumerate(self._rows):
            if type(row.get("source_index")) is not int or row["source_index"] != index:
                raise ValueError("artifact is not one contiguous source prefix.")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id or sample_id in self._sample_ids:
                raise ValueError("index requires unique non-empty sample IDs.")
            self._sample_ids.add(sample_id)
        self._next_source_index = len(self._rows)

    @property
    def completed_samples(self) -> int:
        return self._next_source_index

    def _require_writable(self) -> None:
        if self._lock.closed or self._failed:
            raise RuntimeError("writer is closed or failed; reopen it before resuming.")

    def write_shard(self, payload: Mapping[str, Tensor]) -> tuple[str, str]:
        """Reserve an immutable shard name and durably write its payload."""
        self._require_writable()
        relative = f"shards/{self._next_shard:08d}.pt"
        self._next_shard += 1
        path = self.root / relative
        try:
            atomic_torch_save(path, payload)
            digest = file_sha256(path)
        except BaseException:
            self._failed = True
            raise
        return relative, digest

    def commit(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._require_writable()
        ids: set[str] = set()
        for index, row in enumerate(rows, len(self._rows)):
            if type(row.get("source_index")) is not int or row["source_index"] != index:
                raise ValueError("append must extend the contiguous source prefix.")
            sample_id = row.get("sample_id")
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in ids
                or sample_id in self._sample_ids
            ):
                raise ValueError("append requires unique non-empty sample IDs.")
            ids.add(sample_id)
        try:
            append_jsonl(self.root / "index.jsonl", rows)
        except BaseException:
            self._failed = True
            raise
        self._rows.extend(rows)
        self._sample_ids.update(ids)
        self._next_source_index = len(self._rows)

    def truncate(self, count: int) -> None:
        self._require_writable()
        if type(count) is not int or not 0 <= count <= len(self._rows):
            raise ValueError("recovery prefix must be within the committed index.")
        if count == len(self._rows):
            return
        path = self.root / "index.jsonl"
        try:
            payload = index_bytes(path)
            size = sum(len(line) for line in payload.splitlines(keepends=True)[:count])
            # Hide the suffix before truncating; a crash between these steps is safe.
            atomic_bytes(_marker(path), str(size).encode())
            with path.open("r+b") as stream:
                stream.truncate(size)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            # The durable prefix may already differ from our in-memory rows.
            self._failed = True
            raise
        self._rows = self._rows[:count]
        self._reset()

    def close(self) -> None:
        if not self._lock.closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
