"""Opt-in LongCat boundary diagnostics.

CUDA tracing synchronizes and resets peak memory statistics for the measured
encode call. Keep this instrumentation separate from codec tensor operations.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from torch import Tensor


def start(
    audio: Tensor,
    sample_rate: int,
    device: torch.device,
) -> tuple[Path, float, torch.device | None, dict[str, object]] | None:
    raw_path = os.environ.get("ANYTRAIN_LONGCAT_DEBUG_LOG")
    if not raw_path:
        return None
    cuda_device = device if device.type == "cuda" else None
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
    fields: dict[str, object] = {
        "audio_batch": int(audio.size(0)),
        "audio_dtype": str(audio.dtype),
        "audio_numel": int(audio.numel()),
        "audio_sample_rate": sample_rate,
        "audio_shape": tuple(int(value) for value in audio.shape),
        "audio_seconds": float(audio.size(-1) * audio.size(0) / sample_rate),
        "device": str(device),
        "event": "longcat_encode_boundary",
    }
    return Path(raw_path).expanduser(), time.perf_counter(), cuda_device, fields


def finish(
    trace: tuple[Path, float, torch.device | None, dict[str, object]] | None,
    *,
    extra: dict[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    if trace is None:
        return
    path, started, cuda_device, fields = trace
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
        fields.update(
            {
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(cuda_device)),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(cuda_device)),
                "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(cuda_device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(cuda_device)),
            }
        )
    fields["elapsed_seconds"] = time.perf_counter() - started
    if error is None:
        fields["status"] = "complete"
        if extra:
            fields.update(extra)
    else:
        fields.update(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "error",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(fields, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def shape(value: object) -> object:
    if isinstance(value, Tensor):
        return tuple(int(size) for size in value.shape)
    if isinstance(value, (tuple, list)):
        return [shape(item) for item in value]
    return type(value).__name__


def length_summary(value: Tensor) -> dict[str, int]:
    return {
        "count": int(value.numel()),
        "maximum": int(value.max().item()),
        "minimum": int(value.min().item()),
        "nonzero": int(torch.count_nonzero(value).item()),
    }
