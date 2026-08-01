from __future__ import annotations

from enum import auto

from anytrain._compat import StrEnum


class QuantizerType(StrEnum):
    AGRVQ = auto()
    FSQ = auto()
    GVQ = auto()
    RVQ = auto()
    RFSQ = auto()
    VQ = auto()
