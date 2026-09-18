"""Shared host transfer and memory budgeting for the two expert dispatch paths."""

from torch import Tensor

_LINEAR_ROUTE_WEIGHT_BUDGET_BYTES = 64 * 1024 * 1024


def _host_int_list(values: Tensor) -> list[int]:
    """Materialize a short device count vector for a Python dispatch loop.

    Generic experts have arbitrary Python ``forward`` methods, so their
    variable-length segments still require concrete host bounds.  Keep the
    unavoidable accelerator synchronization to one explicit transfer, while
    making the transfer boundary explicit. CPU values stay on-device and
    incur only ordinary scalar conversion.
    """

    if values.device.type == "cpu":
        # Convert the short vector in one native call.  Scalar-tensor
        # iteration is noticeably slower for larger expert banks and does not
        # avoid any CPU synchronization.
        return values.tolist()
    host_values = values.detach().to(device="cpu")
    return host_values.tolist()


def _linear_route_memory_fits(
    route_count: int,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    stacked_expert_count: int = 0,
    budget_bytes: int = _LINEAR_ROUTE_WEIGHT_BUDGET_BYTES,
) -> bool:
    """Return whether route-parameter temporaries fit the fast-path budget.

    The count comes from tensor shape metadata, so this check does not read a
    CUDA scalar or introduce a synchronization.  Bank execution gathers only
    weights and therefore leaves ``stacked_expert_count`` at zero; ``bias`` is
    accepted for parity with the sparse dispatcher and keeps the helper useful
    for callers that gather both tensors.
    """

    parameter_bytes = weight.numel() * weight.element_size()
    bytes_required = (route_count + stacked_expert_count) * parameter_bytes
    if bias is not None:
        bias_bytes = bias.numel() * bias.element_size()
        bytes_required += (route_count + stacked_expert_count) * bias_bytes
    return bytes_required <= budget_bytes
