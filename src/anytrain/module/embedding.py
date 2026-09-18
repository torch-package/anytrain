"""Reusable parameter owners and directional embedding views.

``EmbeddingBank`` owns trainable modules exactly once. ``EmbeddingView`` is a
non-owning routed view over that bank, so input and output directions can share
selected rows without duplicate optimizer or checkpoint paths.  The module is
domain agnostic: callers provide token ranges, owner maps and optional metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._embedding_contract import owner_contract_state

_SAFE_MODULE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
_MODULE_NAME_PREFIX = "__embedding_owner__"


class EmbeddingBank(nn.Module):
    """Canonical parameter owners for one or more embedding views."""

    def __init__(
        self,
        embedding_dim: int | None = None,
        *,
        grammar: str = "embedding-bank-v1",
        module_name_prefix: str = _MODULE_NAME_PREFIX,
    ) -> None:
        super().__init__()
        if embedding_dim is not None:
            _positive_int(embedding_dim, "embedding dimension")
        if not isinstance(grammar, str) or not grammar:
            raise ValueError("embedding bank grammar must be a non-empty string.")
        if not isinstance(module_name_prefix, str) or not module_name_prefix:
            raise ValueError("embedding bank module name prefix must be non-empty.")
        self.embedding_dim = embedding_dim
        self.grammar = grammar
        self._module_name_prefix = module_name_prefix
        self.owners = nn.ModuleDict()
        self._logical_to_module: dict[str, str] = {}
        self._owner_state: dict[str, dict[str, object]] = {}

    @property
    def owner_names(self) -> tuple[str, ...]:
        return tuple(self.owners.keys())

    def owner(self, name: str) -> nn.Module:
        module_name = self._logical_to_module.get(name, name)
        if module_name not in self.owners:
            raise KeyError(f"unknown embedding owner {name!r}.")
        return self.owners[module_name]

    def owner_state(self, name: str) -> Mapping[str, object]:
        module_name = self._logical_to_module.get(name, name)
        if module_name not in self._owner_state:
            raise KeyError(f"unknown embedding owner {name!r}.")
        return dict(self._owner_state[module_name])

    def add_owner(
        self,
        name: str,
        module: nn.Module,
        *,
        logical_names: Sequence[str] = (),
        state: Mapping[str, object] | None = None,
    ) -> str:
        """Register one canonical owner and optional logical aliases.

        Owners may provide ``embedding_contract_state() -> Mapping[str, object]``
        for their complete structural identity. ``state`` overrides its defaults.
        Owners without that method retain legacy structural metadata inference.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("embedding owner name must be non-empty.")
        if not isinstance(module, nn.Module):
            raise TypeError("embedding owner must be a torch module.")
        module_name = _module_name(name, prefix=self._module_name_prefix)
        if module_name in self.owners:
            raise ValueError(f"embedding owner name {name!r} is already registered.")
        for existing_name, existing in self.owners.items():
            if existing is module:
                raise ValueError(
                    f"embedding module is already registered as owner {existing_name!r}; "
                    "use logical_names for aliases."
                )
        aliases = (name, *logical_names)
        for logical in aliases:
            if not isinstance(logical, str) or not logical:
                raise ValueError("embedding logical owner names must be non-empty.")
            previous = self._logical_to_module.get(logical)
            if previous is not None and previous != module_name:
                raise ValueError(f"embedding logical owner {logical!r} maps to two modules.")
        self.owners[module_name] = module
        for logical in aliases:
            self._logical_to_module[logical] = module_name
        self._owner_state[module_name] = dict(state or {})
        return module_name

    def contract_state(self) -> dict[str, object]:
        owners = {
            name: owner_contract_state(owner, self._owner_state.get(name, {}))
            for name, owner in self.owners.items()
        }
        return {
            "grammar": self.grammar,
            "embedding_dim": self.embedding_dim,
            "logical_to_owner": dict(self._logical_to_module),
            "owners": owners,
        }


class EmbeddingView(nn.Module):
    """Directional routed view over an :class:`EmbeddingBank`.

    ``owner_indices`` and ``owner_rows`` map local vocabulary ids to canonical
    bank owners.  The optional prediction map lets output rows be tied or
    untied independently from input rows.
    """

    is_composite: bool = True
    is_routed: bool = True

    def __init__(
        self,
        bank: EmbeddingBank,
        *,
        direction: str,
        num_embeddings: int,
        embedding_dim: int,
        stream_names: Sequence[str],
        stream_token_ranges: Mapping[str, tuple[tuple[int, int], ...]],
        stream_owner_names: Mapping[str, str],
        owner_names: Sequence[str],
        owner_indices: Tensor,
        owner_rows: Tensor,
        prediction_indices: Tensor | None = None,
        prediction_rows: Tensor | None = None,
        head_tied_by_stream: Mapping[str, bool] | None = None,
        families: Mapping[str, object] | None = None,
        initializations: Mapping[str, object] | None = None,
        tie_groups: Mapping[str, str | None] | None = None,
        contract_family: str = "routed-v1",
        contract_grammar: str = "embedding-view-v1",
    ) -> None:
        super().__init__()
        if not isinstance(bank, EmbeddingBank):
            raise TypeError("embedding view requires an EmbeddingBank.")
        _positive_int(num_embeddings, "embedding view rows")
        _positive_int(embedding_dim, "embedding view dimension")
        _map_tensor(owner_indices, num_embeddings, "owner indices")
        _map_tensor(owner_rows, num_embeddings, "owner rows")
        if (prediction_indices is None) != (prediction_rows is None):
            raise ValueError("prediction owner indices and rows must be provided together.")
        if prediction_indices is not None and prediction_rows is not None:
            _map_tensor(prediction_indices, num_embeddings, "prediction owner indices")
            _map_tensor(prediction_rows, num_embeddings, "prediction owner rows")
        if not isinstance(direction, str) or not direction:
            raise ValueError("embedding view direction must be non-empty.")
        if not isinstance(contract_family, str) or not contract_family:
            raise ValueError("embedding view contract family must be non-empty.")
        if not isinstance(contract_grammar, str) or not contract_grammar:
            raise ValueError("embedding view contract grammar must be non-empty.")

        # Keep the bank alive without registering it as a child.  The model root
        # owns the bank, so tied views never duplicate checkpoint paths.
        object.__setattr__(self, "_bank_owner", bank)
        self.direction = direction
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._stream_names = tuple(stream_names)
        self._stream_token_ranges = {
            name: tuple((bounds[0], bounds[1]) for bounds in ranges)
            for name, ranges in stream_token_ranges.items()
        }
        self._stream_owner_names = dict(stream_owner_names)
        self._owner_names = tuple(owner_names)
        self._head_tied_by_stream = dict(head_tied_by_stream or {})
        self._families = dict(families or {})
        self._initializations = dict(initializations or {})
        self._tie_groups = dict(tie_groups or {})
        self._contract_family = contract_family
        self._contract_grammar = contract_grammar
        self.owner_indices = nn.Buffer(owner_indices.to(dtype=torch.int64), persistent=True)
        self.owner_rows = nn.Buffer(owner_rows.to(dtype=torch.int64), persistent=True)
        if prediction_indices is None:
            self.prediction_indices = None
            self.prediction_row_offsets = None
        else:
            assert prediction_rows is not None
            self.prediction_indices = nn.Buffer(
                prediction_indices.to(dtype=torch.int64), persistent=True
            )
            self.prediction_row_offsets = nn.Buffer(
                prediction_rows.to(dtype=torch.int64), persistent=True
            )

    def _bank(self) -> EmbeddingBank:
        bank = self._bank_owner
        if not isinstance(bank, EmbeddingBank):
            raise RuntimeError("embedding view has no canonical bank owner.")
        return bank

    def _apply(self, fn: Any, recurse: bool = True) -> EmbeddingView:
        self._bank()._apply(fn, recurse=recurse)
        super()._apply(fn, recurse=recurse)
        return self

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def stream_names(self) -> tuple[str, ...]:
        return self._stream_names

    @property
    def stream_token_ranges(self) -> Mapping[str, tuple[tuple[int, int], ...]]:
        return dict(self._stream_token_ranges)

    @property
    def stream_owner_names(self) -> Mapping[str, str]:
        return dict(self._stream_owner_names)

    @property
    def owner(self) -> EmbeddingBank:
        return self._bank()

    @property
    def bank(self) -> EmbeddingBank:
        return self._bank()

    def owner_parameter(self, stream: str) -> nn.Module:
        try:
            name = self._stream_owner_names[stream]
        except KeyError as error:
            raise KeyError(f"unknown embedding stream {stream!r}.") from error
        return self._bank().owner(name)

    @property
    def owner_names(self) -> tuple[str, ...]:
        return self._owner_names

    @property
    def head_tied_by_stream(self) -> Mapping[str, bool]:
        return dict(self._head_tied_by_stream)

    @property
    def head_tied(self) -> bool:
        return all(self._head_tied_by_stream.values())

    @property
    def families(self) -> Mapping[str, object]:
        return dict(self._families)

    @property
    def initializations(self) -> Mapping[str, object]:
        return dict(self._initializations)

    @property
    def tie_groups(self) -> Mapping[str, str | None]:
        return dict(self._tie_groups)

    @property
    def stream_configs(self) -> Mapping[str, Mapping[str, object]]:
        return {
            stream: {
                "family": _value(self._families[stream]),
                "initialization": _value(self._initializations[stream]),
                "head_tied": self._head_tied_by_stream.get(stream, True),
                "tie_group": self._tie_groups.get(stream),
                "rows": sum(end - start for start, end in self._stream_token_ranges[stream]),
                "owner": self._stream_owner_names.get(stream),
            }
            for stream in self._stream_names
            if stream in self._families and stream in self._initializations
        }

    @property
    def prediction_owner_names(self) -> Mapping[str, str]:
        if self.prediction_indices is None:
            return dict(self._stream_owner_names)
        return {
            stream: (self._stream_owner_names[stream] if tied else f"output_prediction_{stream}")
            for stream, tied in self._head_tied_by_stream.items()
        }

    @property
    def family(self) -> object | None:
        values = set(self._families.values())
        return next(iter(values)) if len(values) == 1 else None

    @property
    def initialization(self) -> object | None:
        values = set(self._initializations.values())
        return next(iter(values)) if len(values) == 1 else None

    @property
    def weight(self) -> Tensor:
        return self.rows(torch.arange(self.num_embeddings, device=self.owner_indices.device))

    @property
    def prediction_weight(self) -> Tensor:
        return self.prediction_rows()

    @property
    def effective_weight(self) -> Tensor:
        return self.prediction_rows()

    def rows(self, local_ids: Tensor) -> Tensor:
        ids = _local_ids(local_ids, self.num_embeddings)
        return self._gather(self.owner_indices, self.owner_rows, ids)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.rows(input_ids)

    def prediction_rows(self, local_ids: Tensor | None = None) -> Tensor:
        ids = (
            torch.arange(self.num_embeddings, device=self.owner_indices.device)
            if local_ids is None
            else _local_ids(local_ids, self.num_embeddings)
        )
        if self.prediction_indices is None or self.prediction_row_offsets is None:
            return self._gather(self.owner_indices, self.owner_rows, ids)
        return self._gather(self.prediction_indices, self.prediction_row_offsets, ids)

    def effective_rows(self, local_ids: Tensor | None = None) -> Tensor:
        return self.prediction_rows(local_ids)

    def logit_rows(self, local_ids: Tensor | None = None) -> Tensor:
        return self.prediction_rows(local_ids)

    def output_rows(self, local_ids: Tensor | None = None) -> Tensor:
        return self.prediction_rows(local_ids)

    def fused_linear_weight(self) -> Tensor:
        return self.prediction_rows()

    def logits(self, hidden: Tensor, local_ids: Tensor | None = None) -> Tensor:
        rows = self.prediction_rows(local_ids)
        return F.linear(hidden.to(dtype=rows.dtype), rows)

    def contract_state(self) -> dict[str, object]:
        return {
            "family": self._contract_family,
            "grammar": self._contract_grammar,
            "direction": self.direction,
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "streams": list(self.stream_names),
            "stream_token_ranges": {
                name: [list(bounds) for bounds in ranges]
                for name, ranges in self._stream_token_ranges.items()
            },
            "stream_owner_names": dict(self._stream_owner_names),
            "stream_configs": {name: dict(config) for name, config in self.stream_configs.items()},
            "owner_names": list(self.owner_names),
            "prediction_owner_names": dict(self.prediction_owner_names),
            "head_tied": self.head_tied,
            "head_tied_by_stream": dict(self._head_tied_by_stream),
            "families": {name: _value(value) for name, value in self._families.items()},
            "initializations": {
                name: _value(value) for name, value in self._initializations.items()
            },
            "tie_groups": dict(self._tie_groups),
            "prediction_untied": self.prediction_indices is not None,
        }

    def _gather(self, owner_indices: Tensor, row_offsets: Tensor, ids: Tensor) -> Tensor:
        bank = self._bank()
        target_device = _module_device(bank, fallback=owner_indices.device)
        flat_ids = ids.reshape(-1).to(device=owner_indices.device, dtype=torch.long)
        if flat_ids.numel() == 0:
            return torch.empty(
                (*ids.shape, self.embedding_dim),
                device=target_device,
                dtype=_module_dtype(bank),
            )
        owner_ids = owner_indices.index_select(0, flat_ids)
        rows = row_offsets.to(device=owner_indices.device).index_select(0, flat_ids)
        result: Tensor | None = None
        for owner_index, name in enumerate(self._owner_names):
            positions = (owner_ids == owner_index).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            owner = bank.owner(name)
            values = _owner_rows(owner, rows.index_select(0, positions)).to(device=target_device)
            if result is None:
                result = values.new_empty(flat_ids.numel(), values.size(-1))
            result = result.index_copy(
                0,
                positions.to(device=result.device),
                values.to(dtype=result.dtype),
            )
        if result is None:
            raise RuntimeError("embedding view has no routed owners.")
        return result.view(*ids.shape, self.embedding_dim)


def _map_tensor(value: Tensor, size: int, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if value.dtype not in {torch.int32, torch.int64}:
        raise TypeError(f"{name} must use int32 or int64 dtype.")


def _owner_rows(owner: nn.Module, rows: Tensor) -> Tensor:
    parameter = next(owner.parameters(), None)
    if parameter is not None:
        rows = rows.to(device=parameter.device)
    method = getattr(owner, "rows", None)
    value = method(rows) if callable(method) else owner(rows)
    if not isinstance(value, Tensor):
        raise TypeError("embedding owner rows must return a Tensor.")
    return value


def _module_device(module: nn.Module, *, fallback: torch.device) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    return fallback if buffer is None else buffer.device


def _module_dtype(module: nn.Module) -> torch.dtype:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(), None)
    if buffer is not None and (torch.is_floating_point(buffer) or buffer.is_complex()):
        return buffer.dtype
    return torch.get_default_dtype()


def _local_ids(ids: Tensor, size: int) -> Tensor:
    if not isinstance(ids, Tensor):
        raise TypeError("embedding IDs must be a Tensor.")
    if ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("embedding IDs must use int32 or int64 dtype.")
    if bool((ids < 0).any()) or bool((ids >= size).any()):
        raise IndexError("embedding ID is out of range.")
    return ids.to(dtype=torch.long)


def _module_name(name: str, *, prefix: str) -> str:
    safe = _SAFE_MODULE_NAME_RE.fullmatch(name) is not None
    if safe and not name.startswith(prefix):
        return name
    tag = "s" if safe else "u"
    return f"{prefix}{tag}{name.encode('utf-8').hex()}"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _value(value: object) -> object:
    return getattr(value, "value", value)


__all__ = ["EmbeddingBank", "EmbeddingView"]
