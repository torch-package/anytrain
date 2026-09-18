"""Embedding owner contracts and compatibility with legacy structural metadata.

New owners implement embedding_contract_state(); their complete payload avoids
attribute probing (including expensive dense weight properties). Explicit bank
metadata overrides provider defaults. Legacy owners retain their exact payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

from torch import Tensor, nn

from anytrain._contract import canonical_sha256, canonical_value


def owner_contract_state(owner: nn.Module, metadata: Mapping[str, object]) -> dict[str, object]:
    provider = getattr(owner, "embedding_contract_state", None)
    if provider is None:
        result = _legacy_owner_contract_state(owner, metadata)
    else:
        if not callable(provider):
            raise TypeError("embedding_contract_state must be callable.")
        provided = provider()
        if not isinstance(provided, Mapping):
            raise TypeError("embedding_contract_state() must return a mapping.")
        result = {**provided, **metadata}
    canonical = canonical_value(result)
    if not isinstance(canonical, dict):
        raise TypeError("embedding owner contract state must be a mapping.")
    return canonical


def _legacy_owner_contract_state(
    owner: nn.Module,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = dict(metadata)
    implementation = result.get("implementation")
    if implementation is None:
        implementation = (
            "dense-v1"
            if isinstance(owner, nn.Embedding)
            else f"{type(owner).__module__}.{type(owner).__qualname__}"
        )
        # Factorized owners conventionally expose these structural attributes;
        # keep their identity visible without importing a domain implementation.
        if hasattr(owner, "fsq_levels") and hasattr(owner, "codebook_sizes"):
            implementation = "fsq-factorized-v1"
        result["implementation"] = implementation
    if isinstance(owner, nn.Embedding):
        result.setdefault("family", "dense")
        result.setdefault("rows", owner.num_embeddings)
        result.setdefault("embedding_dim", owner.embedding_dim)
    else:
        rows = getattr(owner, "num_embeddings", None)
        dim = getattr(owner, "embedding_dim", None)
        if isinstance(rows, Integral) and not isinstance(rows, bool):
            result.setdefault("rows", int(rows))
        if isinstance(dim, Integral) and not isinstance(dim, bool):
            result.setdefault("embedding_dim", int(dim))
        # Factorized owners may expose ``weight`` as an expensive dense
        # compatibility property. Their structural attributes above are
        # sufficient for the contract, so avoid materializing that table.
        if not (hasattr(owner, "fsq_levels") and hasattr(owner, "codebook_sizes")):
            weight = getattr(owner, "weight", None)
            if isinstance(weight, Tensor) and weight.ndim >= 2:
                result.setdefault("rows", int(weight.shape[0]))
                result.setdefault("embedding_dim", int(weight.shape[-1]))

    config = getattr(owner, "config", None)
    feature = getattr(config, "feature", None)
    if feature is not None:
        result.setdefault("feature", _value(feature))
    for attribute in ("codebook_sizes", "fsq_levels", "radix_order"):
        value = getattr(owner, attribute, None)
        if value is not None:
            result.setdefault(attribute, _json_value(value))
    topology = getattr(owner, "topology", None)
    if isinstance(topology, Tensor):
        result.setdefault("topology", topology.detach().cpu().tolist())
    level_values = getattr(owner, "level_values", None)
    if isinstance(level_values, Tensor):
        result.setdefault(
            "level_values_sha256",
            canonical_sha256(level_values.detach().cpu().tolist()),
        )
    return result


def _value(value: object) -> object:
    return getattr(value, "value", value)


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return _value(value)
