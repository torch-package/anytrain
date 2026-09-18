"""Recognition of exact adapter types and FSDP2 dynamic wrappers."""
from typing import TypeGuard, TypeVar

from torch import nn

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def is_exact_or_fsdp(module: object, expected: type[_ModuleT]) -> TypeGuard[_ModuleT]:
    if type(module) is expected:
        return True
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:  # pragma: no cover - torch>=2.8 exposes FSDPModule.
        return False
    return isinstance(module, FSDPModule) and isinstance(module, expected)
