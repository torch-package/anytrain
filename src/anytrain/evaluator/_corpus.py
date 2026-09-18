"""Root-only corpus computation with an error envelope observed by every rank."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch.distributed as dist

from anytrain._audit import audit_collective

_Payload = TypeVar("_Payload")


def compute_on_root(
    payload: _Payload,
    compute: Callable[[Sequence[_Payload | None]], dict[str, object]],
    *,
    source: str,
    label: str,
    invalid_envelope: str,
    local_item_count: int,
) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Torch's object collective mutates a heterogeneous list; domain payload
    # validation belongs to the evaluator's merge function on rank zero.
    gathered: list[Any] | None = [None] * world_size if rank == 0 else None
    audit_collective(
        "torch.distributed.gather_object",
        lambda: dist.gather_object(payload, gathered, dst=0),
        source=f"{source}.gather",
        object_type=type(payload).__name__,
        local_item_count=local_item_count,
        world_size=world_size,
    )
    envelope: list[object] = [None]
    root_error: Exception | None = None
    if rank == 0:
        try:
            if gathered is None:
                raise RuntimeError(f"Distributed {label} root did not allocate gather storage.")
            envelope[0] = {"status": "ok", **compute(gathered)}
        except Exception as error:
            root_error = error
            envelope[0] = {
                "status": "error", "error_type": type(error).__name__, "error": str(error)
            }
    audit_collective(
        "torch.distributed.broadcast_object_list",
        lambda: dist.broadcast_object_list(envelope, src=0),
        source=f"{source}.broadcast",
        object_type="dict",
        world_size=world_size,
    )
    result = envelope[0]
    if not isinstance(result, dict) or result.get("status") not in {"ok", "error"}:
        raise RuntimeError(invalid_envelope)
    if result["status"] == "error":
        error_type = result.get("error_type", "Exception")
        error_message = result.get("error", "unknown root error")
        failure = RuntimeError(
            f"Distributed {label} compute failed on rank zero: {error_type}: {error_message}"
        )
        if root_error is not None:
            raise failure from root_error
        raise failure
    return result
