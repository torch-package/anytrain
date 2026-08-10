"""First-class MoE experts: callable cell ops that bind to a routing grid."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from torch import Size, Tensor
from torch.nn import functional as F

from .grid import Grid1d, MoEGrid, NativeGrid, PaddingMode

if TYPE_CHECKING:
    from .bank import ParamExpertBank
    from .partitioner import MoEPartitioner

BiasKind = Literal["none", "additive_channel"]


class ExpertFamily(Enum):
    """Execution family selected at :meth:`Expert.bind` time."""

    LINEAR = auto()
    CONV1D = auto()
    CONV_TRANSPOSE1D = auto()
    CONV2D = auto()
    GENERIC = auto()


@dataclass(eq=False)
class BoundExpert:
    """Internal runtime produced by :meth:`Expert.bind`."""

    bank: ParamExpertBank
    partitioner: MoEPartitioner


@runtime_checkable
class Expert(Protocol):
    """First-class MoE expert: callable cell op + bindable routing geometry."""

    @property
    def bias_kind(self) -> BiasKind:
        """How MoE may treat bias relative to partition/restore."""
        ...

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
    ) -> Tensor:
        """Single-expert op on already-partitioned cells (torch-style)."""
        ...

    def bind(
        self,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        grid: MoEGrid,
    ) -> BoundExpert:
        """Validate shapes and build the internal bank + partitioner pair."""
        ...


# Backward-compatible alias; prefer :class:`Expert`.
ExpertFn = Expert


def _require_bias_kind(
    bias_kind: BiasKind,
    bias: Tensor | None,
) -> None:
    if bias_kind == "none" and bias is not None:
        raise ValueError(
            "Expert.bias_kind is 'none' but bias was provided; "
            "pass bias=None or use bias_kind='additive_channel'."
        )


def _validate_conv_bias(
    bias: Tensor | None,
    *,
    num_experts: int,
    out_channels: int,
) -> None:
    if bias is None:
        return
    if bias.shape != (num_experts, out_channels):
        raise ValueError(
            "conv bias must be shaped (num_experts, out_channels), "
            f"got {tuple(bias.shape)}."
        )


@dataclass(frozen=True)
class CallableExpert:
    """Wrap a plain cell fn for native-resolution MoE only."""

    fn: Callable[[Tensor, Tensor, Tensor | None], Tensor]
    bias_kind: BiasKind = "none"

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
    ) -> Tensor:
        return self.fn(input, weight, bias)

    def bind(
        self,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        grid: MoEGrid,
    ) -> BoundExpert:
        from .bank import ParamExpertBank
        from .partitioner import IdentityPartitioner
        from .routing import validate_num_experts

        validate_num_experts(num_experts)
        _require_bias_kind(self.bias_kind, bias)
        if not isinstance(grid, NativeGrid):
            raise TypeError(
                "CallableExpert only supports NativeGrid / grid=None; "
                f"got grid={type(grid)!r}."
            )
        bank = ParamExpertBank(
            self,
            weight,
            bias,
            num_experts=num_experts,
            family=ExpertFamily.GENERIC,
            bias_kind=self.bias_kind,
        )
        return BoundExpert(bank=bank, partitioner=IdentityPartitioner())


@dataclass(frozen=True)
class LinearExpert:
    """``F.linear`` expert; native-resolution grid only."""

    @property
    def bias_kind(self) -> BiasKind:
        return "additive_channel"

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
    ) -> Tensor:
        return F.linear(input, weight, bias)

    def bind(
        self,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        grid: MoEGrid,
    ) -> BoundExpert:
        from .bank import ParamExpertBank
        from .partitioner import IdentityPartitioner
        from .routing import validate_num_experts

        validate_num_experts(num_experts)
        _require_bias_kind(self.bias_kind, bias)
        if not isinstance(grid, NativeGrid):
            raise TypeError("linear experts only support NativeGrid / grid=None.")
        if weight.ndim != 3:
            raise ValueError(
                "linear weight must be shaped (num_experts, out_features, in_features), "
                f"got {tuple(weight.shape)}."
            )
        if bias is not None and bias.shape != (num_experts, weight.size(1)):
            raise ValueError(
                "linear bias must be shaped (num_experts, out_features), "
                f"got {tuple(bias.shape)}."
            )
        bank = ParamExpertBank(
            self,
            weight,
            bias,
            num_experts=num_experts,
            family=ExpertFamily.LINEAR,
            bias_kind=self.bias_kind,
        )
        return BoundExpert(bank=bank, partitioner=IdentityPartitioner())


@dataclass(frozen=True)
class Conv1dExpert:
    """``F.conv1d`` expert; geometry drives concat-grid compilation."""

    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1
    causal: bool = False
    padding_mode: PaddingMode = "zeros"

    def __post_init__(self) -> None:
        if self.groups <= 0:
            raise ValueError(f"groups must be positive, got {self.groups}.")
        if self.padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(f"unsupported padding_mode {self.padding_mode!r}.")

    @property
    def bias_kind(self) -> BiasKind:
        return "additive_channel"

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
    ) -> Tensor:
        return F.conv1d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def bind(
        self,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        grid: MoEGrid,
    ) -> BoundExpert:
        from .bank import ParamExpertBank, with_execution_padding
        from .conv.bank import validate_channels
        from .conv.partitioner import Conv1dGridPartitioner, conv1d_execution_padding
        from .routing import validate_num_experts

        validate_num_experts(num_experts)
        _require_bias_kind(self.bias_kind, bias)
        if weight.ndim != 4:
            raise ValueError(
                "Conv1dExpert weight must be shaped "
                "(num_experts, out_channels, in_channels // groups, kernel_size), "
                f"got {tuple(weight.shape)}."
            )
        out_channels = int(weight.size(1))
        in_channels = int(weight.size(2)) * self.groups
        kernel_size = int(weight.size(3))
        validate_channels(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=self.groups,
            num_experts=num_experts,
        )
        _validate_conv_bias(bias, num_experts=num_experts, out_channels=out_channels)

        context = "global"
        cell_size: int | None = None
        fill = self.padding_mode
        if isinstance(grid, Grid1d):
            cell_size = grid.segment_size
            context = "global" if grid.context is None else grid.context
            fill = self.padding_mode if grid.fill is None else grid.fill
        elif not isinstance(grid, NativeGrid):
            raise TypeError(f"unsupported grid for Conv1dExpert: {type(grid)!r}.")
        execution_padding = conv1d_execution_padding(
            cell_size=None if cell_size is None else Size([cell_size]),
            context=context,
            causal=self.causal,
            padding=self.padding,
        )
        bank_expert = with_execution_padding(self, execution_padding=execution_padding)
        bank = ParamExpertBank(
            bank_expert,
            weight,
            bias,
            num_experts=num_experts,
            family=ExpertFamily.CONV1D,
            bias_kind=self.bias_kind,
        )
        partitioner = Conv1dGridPartitioner(
            kernel_size=kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            fill=fill,
            cell_size=cell_size,
            causal=self.causal,
            context=context,
            in_channels=in_channels,
        )
        return BoundExpert(bank=bank, partitioner=partitioner)


@dataclass(frozen=True)
class ConvTranspose1dExpert:
    """``F.conv_transpose1d`` expert; geometry drives overlap-add grid compilation."""

    stride: int = 1
    padding: int = 0
    output_padding: int = 0
    dilation: int = 1
    groups: int = 1

    def __post_init__(self) -> None:
        if self.groups <= 0:
            raise ValueError(f"groups must be positive, got {self.groups}.")

    @property
    def bias_kind(self) -> BiasKind:
        return "additive_channel"

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
    ) -> Tensor:
        return F.conv_transpose1d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def bind(
        self,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        grid: MoEGrid,
    ) -> BoundExpert:
        from .bank import ParamExpertBank
        from .conv.bank import validate_channels
        from .conv.partitioner import ConvTranspose1dGridPartitioner
        from .routing import validate_num_experts

        validate_num_experts(num_experts)
        _require_bias_kind(self.bias_kind, bias)
        if weight.ndim != 4:
            raise ValueError(
                "ConvTranspose1dExpert weight must be shaped "
                "(num_experts, in_channels, out_channels // groups, kernel_size), "
                f"got {tuple(weight.shape)}."
            )
        in_channels = int(weight.size(1))
        out_channels = int(weight.size(2)) * self.groups
        kernel_size = int(weight.size(3))
        validate_channels(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=self.groups,
            num_experts=num_experts,
        )
        _validate_conv_bias(bias, num_experts=num_experts, out_channels=out_channels)

        cell_size: int | None = None
        if isinstance(grid, Grid1d):
            context = "global" if grid.context is None else grid.context
            fill = "zeros" if grid.fill is None else grid.fill
            if context != "global" or fill != "zeros":
                raise ValueError(
                    "ConvTranspose1dExpert only uses Grid1d.segment_size; "
                    "context must be 'global' and fill must be 'zeros', "
                    f"got context={context!r}, fill={fill!r}."
                )
            cell_size = grid.segment_size
        elif not isinstance(grid, NativeGrid):
            raise TypeError(
                f"unsupported grid for ConvTranspose1dExpert: {type(grid)!r}."
            )

        bank = ParamExpertBank(
            self,
            weight,
            bias,
            num_experts=num_experts,
            family=ExpertFamily.CONV_TRANSPOSE1D,
            bias_kind=self.bias_kind,
        )
        partitioner = ConvTranspose1dGridPartitioner(
            kernel_size=kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            output_padding=self.output_padding,
            cell_size=cell_size,
            in_channels=in_channels,
        )
        return BoundExpert(bank=bank, partitioner=partitioner)


@dataclass(frozen=True)
class Conv2dExpert:
    """``F.conv2d`` expert; native-resolution grid only."""

    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1

    def __post_init__(self) -> None:
        if self.groups <= 0:
            raise ValueError(f"groups must be positive, got {self.groups}.")
        for name, value in (
            ("stride", self.stride),
            ("padding", self.padding),
            ("dilation", self.dilation),
        ):
            if len(value) != 2:
                raise ValueError(f"{name} must be a length-2 tuple, got {value!r}.")

    @property
    def bias_kind(self) -> BiasKind:
        return "additive_channel"

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        /,
    ) -> Tensor:
        return F.conv2d(
            input,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def bind(
        self,
        weight: Tensor,
        bias: Tensor | None,
        *,
        num_experts: int,
        grid: MoEGrid,
    ) -> BoundExpert:
        from .bank import ParamExpertBank
        from .conv.bank import validate_channels
        from .conv.partitioner import Conv2dIdentityPartitioner
        from .routing import validate_num_experts

        validate_num_experts(num_experts)
        _require_bias_kind(self.bias_kind, bias)
        if not isinstance(grid, NativeGrid):
            raise TypeError("Conv2dExpert only supports NativeGrid / grid=None.")
        if weight.ndim != 5:
            raise ValueError(
                "Conv2dExpert weight must be shaped "
                "(num_experts, out_channels, in_channels // groups, kH, kW), "
                f"got {tuple(weight.shape)}."
            )
        out_channels = int(weight.size(1))
        in_channels = int(weight.size(2)) * self.groups
        validate_channels(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=self.groups,
            num_experts=num_experts,
        )
        _validate_conv_bias(bias, num_experts=num_experts, out_channels=out_channels)
        bank = ParamExpertBank(
            self,
            weight,
            bias,
            num_experts=num_experts,
            family=ExpertFamily.CONV2D,
            bias_kind=self.bias_kind,
        )
        return BoundExpert(
            bank=bank,
            partitioner=Conv2dIdentityPartitioner(in_channels=in_channels),
        )


linear = LinearExpert()

__all__ = [
    "BiasKind",
    "BoundExpert",
    "CallableExpert",
    "Conv1dExpert",
    "Conv2dExpert",
    "ConvTranspose1dExpert",
    "Expert",
    "ExpertFamily",
    "ExpertFn",
    "LinearExpert",
    "linear",
]
