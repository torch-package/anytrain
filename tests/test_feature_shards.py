"""Committed-index visibility and restart tests; no model or GPU required."""

from __future__ import annotations

import pytest
import torch

from anytrain.audio.codec import shards


def test_failed_watermark_hides_tail_and_retry_discards_it(tmp_path, monkeypatch):
    root = tmp_path / "store"
    with shards.ShardWriter(root) as writer:
        writer.commit([{"source_index": 0, "sample_id": "a"}])

        def fail(path, value):
            raise OSError("commit interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(shards, "atomic_bytes", fail)
            with pytest.raises(OSError):
                writer.commit([{"source_index": 1, "sample_id": "unpublished"}])
        assert [row["sample_id"] for row in shards.read_index(root / "index.jsonl")] == ["a"]
    with shards.ShardWriter(root) as writer:
        writer.commit([{"source_index": 1, "sample_id": "b"}])
    assert [row["sample_id"] for row in shards.read_index(root / "index.jsonl")] == ["a", "b"]


def test_legacy_index_and_partial_committed_record(tmp_path):
    path = tmp_path / "index.jsonl"
    path.write_bytes(b'{"source_index":0,"sample_id":"a"}\n')
    with shards.ShardWriter(tmp_path) as writer:
        writer.commit([{"source_index": 1, "sample_id": "b"}])
    assert len(shards.read_index(path)) == 2
    path.write_bytes(b"{")
    with pytest.raises(ValueError, match="shorter"):
        shards.read_index(path)


def test_duplicate_ids_rejected_before_commit(tmp_path):
    with shards.ShardWriter(tmp_path) as writer, pytest.raises(ValueError, match="unique"):
        writer.commit(
            [{"source_index": 0, "sample_id": "a"}, {"source_index": 1, "sample_id": "a"}]
        )
    assert shards.read_index(tmp_path / "index.jsonl") == ()


def test_group_recovery_preserves_immutable_shards(tmp_path):
    with shards.ShardWriter(tmp_path) as writer:
        writer.commit(
            [{"source_index": 0, "sample_id": "a"}, {"source_index": 1, "sample_id": "b"}]
        )
        payload = tmp_path / "shards" / "00000000.pt"
        payload.write_bytes(b"immutable")
        writer.truncate(1)
        assert payload.read_bytes() == b"immutable"
    with shards.ShardWriter(tmp_path) as writer:
        assert writer.completed_samples == 1
        writer.commit([{"source_index": 1, "sample_id": "b"}])
    assert len(shards.read_index(tmp_path / "index.jsonl")) == 2


def test_failed_writer_cannot_overwrite_published_shard(tmp_path, monkeypatch):
    with shards.ShardWriter(tmp_path) as writer:
        first, _ = writer.write_shard({"features": torch.ones(1)})
        writer.commit([{"source_index": 0, "sample_id": "a", "shard": first}])
        second, _ = writer.write_shard({"features": torch.zeros(1)})

        def fail(path, value):
            raise OSError("interrupted commit")

        with monkeypatch.context() as patch:
            patch.setattr(shards, "atomic_bytes", fail)
            with pytest.raises(OSError):
                writer.commit([{"source_index": 1, "sample_id": "b", "shard": second}])
        with pytest.raises(RuntimeError, match="reopen"):
            writer.write_shard({"features": torch.full((1,), 2.0)})
    assert torch.load(tmp_path / first, weights_only=True)["features"].item() == 1
    with shards.ShardWriter(tmp_path) as writer:
        third, _ = writer.write_shard({"features": torch.full((1,), 3.0)})
        assert third not in {first, second}
        writer.commit([{"source_index": 1, "sample_id": "b", "shard": third}])
    assert len(shards.read_index(tmp_path / "index.jsonl")) == 2
