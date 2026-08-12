from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TypeAlias


class AlignMethod(StrEnum):
    IDENTITY = auto()
    NEAREST = auto()
    LINEAR = auto()
    POOL = auto()
    BROADCAST = auto()
    ATTENTION = auto()


class AlignMerge(StrEnum):
    ADD = auto()
    CONCAT = auto()
    SOURCE = auto()


@dataclass(frozen=True, slots=True)
class AlignSpec:
    method: AlignMethod
    merge: AlignMerge = AlignMerge.ADD

    def __post_init__(self) -> None:
        if not isinstance(self.method, AlignMethod):
            raise TypeError("align spec method must be an AlignMethod.")
        if not isinstance(self.merge, AlignMerge):
            raise TypeError("align spec merge must be an AlignMerge.")


@dataclass(frozen=True, slots=True)
class Stream:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stream name must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class Align:
    """A deferred relation whose output keeps the carrier timeline."""

    name: str
    carrier: TimelineExpr | str
    source: TimelineExpr | str
    spec: AlignSpec
    operator: str = "default"

    def __post_init__(self) -> None:
        _name(self.name, "align name")
        _name(self.operator, "align operator")
        if not isinstance(self.spec, AlignSpec):
            raise TypeError("align spec must be an AlignSpec.")
        _validate_timeline_ref(self.carrier, "align carrier")
        _validate_timeline_ref(self.source, "align source")


@dataclass(frozen=True, slots=True)
class ControlTokens:
    segment_start: str | None = None
    segment_end: str | None = None
    frame_start: str | None = None
    frame_end: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("segment_start", self.segment_start),
            ("segment_end", self.segment_end),
            ("frame_start", self.frame_start),
            ("frame_end", self.frame_end),
        ):
            if value is not None:
                _name(value, f"control {name}")


@dataclass(frozen=True, slots=True)
class StreamMajor:
    items: tuple[ExprRef, ...]
    controls: ControlTokens = ControlTokens()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("stream-major items must be a tuple.")
        if not isinstance(self.controls, ControlTokens):
            raise TypeError("stream-major controls must be ControlTokens.")
        if not self.items:
            raise ValueError("stream-major segment must contain at least one item.")
        for item in self.items:
            _validate_ref(item, "stream-major item")
        if self.controls.frame_start is not None or self.controls.frame_end is not None:
            raise ValueError("stream-major segments cannot contain frame controls.")


@dataclass(frozen=True, slots=True)
class FrameMajor:
    items: tuple[TimelineExpr | str, ...]
    controls: ControlTokens = ControlTokens()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("frame-major items must be a tuple.")
        if not isinstance(self.controls, ControlTokens):
            raise TypeError("frame-major controls must be ControlTokens.")
        if not self.items:
            raise ValueError("frame-major segment must contain at least one item.")
        for item in self.items:
            _validate_timeline_ref(item, "frame-major item")


FlattenSegment: TypeAlias = StreamMajor | FrameMajor


@dataclass(frozen=True, slots=True)
class Flatten:
    name: str
    segments: tuple[FlattenSegment, ...]

    def __post_init__(self) -> None:
        _name(self.name, "flatten name")
        if not isinstance(self.segments, tuple):
            raise TypeError("flatten segments must be a tuple.")
        if not self.segments:
            raise ValueError("flatten expression must contain at least one segment.")
        if any(not isinstance(segment, (StreamMajor, FrameMajor)) for segment in self.segments):
            raise TypeError("flatten segments must be StreamMajor or FrameMajor.")


TimelineExpr: TypeAlias = Stream | Align
Expr: TypeAlias = TimelineExpr | Flatten
ExprRef: TypeAlias = Expr | str


def _name(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")


def _validate_ref(value: object, field: str) -> None:
    if isinstance(value, str):
        _name(value, field)
    elif not isinstance(value, (Stream, Align, Flatten)):
        raise TypeError(f"{field} must be an expression or stream name.")


def _validate_timeline_ref(value: object, field: str) -> None:
    if isinstance(value, str):
        _name(value, field)
    elif not isinstance(value, (Stream, Align)):
        raise TypeError(f"{field} must be a Stream, Align, or stream name.")


__all__ = [
    "Align",
    "AlignMerge",
    "AlignMethod",
    "AlignSpec",
    "ControlTokens",
    "Expr",
    "Flatten",
    "FlattenSegment",
    "FrameMajor",
    "Stream",
    "StreamMajor",
    "TimelineExpr",
]
