from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

Block = tuple[int, int]


class Layout:
    __slots__ = ("_block_names", "_vocab_size", "blocks")

    def __init__(self, **blocks: Block) -> None:
        if not blocks:
            raise ValueError("Layout must contain at least one id block.")
        self.blocks = MappingProxyType(_blocks(blocks))
        self._block_names = tuple(self.blocks)
        self._vocab_size = max((end for _, end in self.blocks.values()), default=0)

    @property
    def block_names(self) -> tuple[str, ...]:
        return self._block_names

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def block(self, name: str) -> Block:
        try:
            return self.blocks[name]
        except KeyError as error:
            raise KeyError(f"unknown id block {name!r}.") from error

    def block_name_for_id(self, token_id: int) -> str:
        for name, (start, end) in self.blocks.items():
            if start <= token_id < end:
                return name
        raise ValueError(f"token_id is outside all id blocks: {token_id}.")

    def to_global(
        self,
        name: str,
        ids: torch.Tensor,
    ) -> torch.Tensor:
        start, end = self.block(name)
        size = end - start
        out_of_range = (ids < 0) | (ids >= size)
        if bool(out_of_range.any()):
            raise ValueError(f"ids contains local id outside block {name!r}.")
        return ids + start

    def to_local(self, ids: torch.Tensor, *, ignore: int | None = None) -> torch.Tensor:
        if ids.numel() == 0:
            raise ValueError("ids must not be empty.")
        active = ids != ignore if ignore is not None else torch.ones_like(ids, dtype=torch.bool)
        active_ids = ids[active]
        if active_ids.numel() == 0:
            return ids.clone()
        first_id = int(active_ids.reshape(-1)[0].detach().cpu())
        name = self.block_name_for_id(first_id)
        start, end = self.blocks[name]
        inside = (active_ids >= start) & (active_ids < end)
        if not bool(inside.all()):
            raise ValueError("ids must all belong to the same id block.")
        local = ids.clone()
        local[active] = active_ids - start
        return local


def _blocks(blocks: Mapping[str, Block]) -> dict[str, Block]:
    parsed: dict[str, Block] = {}
    for name, block in blocks.items():
        _name(name, name="block name")
        if not isinstance(block, tuple) or len(block) != 2:
            raise TypeError(f"blocks[{name!r}] must be a (start, end) tuple.")
        start, end = block
        if start < 0:
            raise ValueError(f"blocks[{name!r}] start must be non-negative.")
        if end <= start:
            raise ValueError(f"blocks[{name!r}] end must be greater than start.")
        if any(_blocks_overlap((start, end), existing) for existing in parsed.values()):
            raise ValueError("id blocks must not overlap.")
        parsed[name] = (start, end)
    return parsed


def _blocks_overlap(left: Block, right: Block) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _name(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if "." in value:
        raise ValueError(f"{name} must not contain '.'.")


__all__ = [
    "Block",
    "Layout",
]
