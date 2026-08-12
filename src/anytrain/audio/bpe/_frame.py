from __future__ import annotations

import operator
from collections.abc import Sequence
from numbers import Integral

from anytrain._compat import strict_zip

Frame = tuple[int, ...]


class FrameCodec:
    def __init__(self, codebook_sizes: Sequence[int]) -> None:
        self.codebook_sizes = codebook_sizes_tuple(codebook_sizes)
        self.num_codebooks = len(self.codebook_sizes)

        strides: list[int] = []
        stride = 1
        for size in self.codebook_sizes:
            strides.append(stride)
            stride *= size
        self.strides = tuple(strides)
        self.vocab_size = stride

    def encode(self, frame: Sequence[int]) -> int:
        values = frame_tuple(frame, self.codebook_sizes)
        return sum(value * stride for value, stride in strict_zip(values, self.strides))

    def decode(self, base_id: int) -> Frame:
        base_id = frame_id(base_id)
        if base_id < 0 or base_id >= self.vocab_size:
            raise ValueError("frame id is outside the configured codebook space")

        values: list[int] = []
        remaining = base_id
        for size in self.codebook_sizes:
            values.append(remaining % size)
            remaining //= size
        return tuple(values)


def codebook_sizes_tuple(codebook_sizes: Sequence[int]) -> tuple[int, ...]:
    if isinstance(codebook_sizes, (str, bytes)) or not isinstance(codebook_sizes, Sequence):
        raise TypeError("codebook_sizes must be a sequence of positive integers")
    sizes = tuple(codebook_sizes)
    if not sizes:
        raise ValueError("codebook_sizes must not be empty")
    parsed: list[int] = []
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, Integral):
            raise TypeError("codebook sizes must be integers")
        resolved = int(size)
        if resolved <= 0:
            raise ValueError("codebook sizes must be positive")
        parsed.append(resolved)
    return tuple(parsed)


def frame_tuple(frame: Sequence[int], codebook_sizes: Sequence[int]) -> Frame:
    if isinstance(frame, (int, str)):
        raise TypeError("frames must be integer sequences; use [id] for single-codebook frames")

    values = tuple(frame_id(value) for value in frame)
    if len(values) != len(codebook_sizes):
        raise ValueError("frames must match the configured number of codebooks")

    for index, (value, size) in enumerate(strict_zip(values, codebook_sizes)):
        if value < 0 or value >= size:
            raise ValueError(f"frame code id at codebook {index} must be in [0, {size})")
    return values


def frame_id(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("frame code ids must be integers")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError("frame code ids must be integers") from error
