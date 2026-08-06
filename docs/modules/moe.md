# anytrain.module.moe Design

## Positioning

`anytrain.module.moe` is the task-agnostic MoE component family for downstream
LightningModules. The public model is a single `MoE` entry whose routing density
is a runtime execution strategy:

- `top_k=None` means dense mixture over all experts.
- `top_k=K` means sparse top-k mixture over selected experts.
- Dense and top-k execution return the same `MoEOutput` and can be switched at
  runtime when the expert bank supports both paths.

`num_experts` is the single public source of truth on `MoE`. The reliable
construction contract is:

1. an `Expert` object (`__call__` + `bias_kind` + `bind`),
2. expert-major `weight` / `bias` Parameters already shaped `(E, ...)`,
3. an optional thin `grid` descriptor.

MoE calls `expert.bind(weight, bias, num_experts=..., grid=...)` to build the
internal bank and partitioner. It does not accept a module list, and it does
not infer experts from arbitrary `nn.Module` graphs.

`Grid1d` expresses 1D segmentation intent. Its `segment_size` is the logical
owned-region stride used by routing and restore; optional `context` / `fill`
overrides are resolved with expert-provided defaults at bind time. Conv
geometry, required context size, and fold law (concat vs overlap-add) live on
the Expert object. Experts receive already partitioned routing cells; they do
not own router or gate logic.

The package remains a component library: it does not provide model-zoo blocks,
training entrypoints, batch-schema assumptions, expert registries, expert
parallel all-to-all, or hidden logging behavior.

## Forward Contract

```python
cells, router_features, layout = partitioner.partition(x)   # internal
logits = router(router_features)          # or cells when router_features is None
probabilities = gate(logits)              # full distribution; default softmax
# sparse: topk(logits) then softmax(selected_logits)
# dense:  execute with full probabilities
routing = materialize_routing(logits, probabilities, top_k=runtime_top_k, mask=layout.mask)
cell_outputs = expert_bank(cells, routing)  # weight-only Expert / grouped kernels
y = layout.restore(cell_outputs)
y = expert_bank.apply_bias(y, ...)           # additive channel bias after restore
return MoEOutput(hidden_states=y, routing=routing, ...)
```

`Expert.__call__` keeps a torch-style `(input, weight, bias=None)` signature.
When `bias_kind == "additive_channel"`, the bank always runs cells **without**
bias, then applies mixed bias after restore. That assumes bias is an additive
channel term (`y = f(x, W) + b`), which matches `nn.Linear` / `nn.Conv*` /
`nn.ConvTranspose*`. Grid / overlap-add partitions are equivalent for the
linear map `f(x, W)`; deferring the additive `b` keeps that equivalence.
`bias_kind == "none"` rejects a non-`None` bias at bind time.

Component boundaries:

- `Expert`: cell op, op geometry, `bind(grid, weight, bias)` → bank + partitioner,
  and `bias_kind`.
- `grid` (`Grid1d` / `NativeGrid` / `None`): thin routing tiling. Not a second
  copy of conv `kernel/stride/padding`.
- Router: maps router features to logits shaped `(..., num_experts)`.
- MoE: owns `num_experts`, orchestration, optional gating/calibration,
  dense/top-k materialization, and output restoration.
- Internal `ParamExpertBank`: family selected at bind time; weight-only cell
  execution then additive bias after restore when applicable.

## Public API

Unified entry:

```python
from anytrain.module.moe import (
    MoE,
    LinearRouter,
    Grid1d,
    linear,
    Conv1dExpert,
    CallableExpert,
)
from anytrain.module import AdaptiveDirichletTempering
from anytrain.module.moe.conv import ConvRouter1d

# Core path: stock or custom Expert + expert-major Parameters.
moe = MoE(
    num_experts=4,
    router=LinearRouter(input_dim, 4),
    expert=linear,
    weight=nn.Parameter(torch.empty(4, out_dim, input_dim)),
    bias=nn.Parameter(torch.empty(4, out_dim)),
    top_k=2,
)

# Convenience only for stock Linear / Conv1d / Conv2d / ConvTranspose1d:
# infer Expert geometry, expand weights, and build a default router.
# This is not a general Module→Expert compiler.
template = nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)
moe = MoE.from_module(
    template,
    num_experts=num_experts,
    gate=AdaptiveDirichletTempering(num_experts=num_experts),
    grid=Grid1d(segment_size),
    top_k=None,
    init="copy",  # copy | kaiming | empty
)

# Lower-level equivalent when you want to pass router / Expert / params explicitly.
from anytrain.module.moe import expand_expert

expert, weight, bias = expand_expert(
    template,
    num_experts=num_experts,
    init="copy",
)
moe = MoE(
    num_experts=num_experts,
    router=ConvRouter1d(channels, num_experts),
    gate=AdaptiveDirichletTempering(num_experts=num_experts),
    expert=expert,
    weight=weight,
    bias=bias,
    grid=Grid1d(segment_size),
    top_k=None,
)

# Custom cell ops: wrap with CallableExpert (native-resolution grid only).
# Segmented / custom fold: implement Expert.bind yourself.
# Do not pass a custom nn.Module into expand_expert.

output = moe(x)                 # MoEOutput
y = output.hidden_states
routing = output.routing
```

Supporting types and helpers:

- `MoEOutput`, `ExpertRouting`, `materialize_routing`
- `LinearRouter`, `Grid1d`, `NativeGrid`
- `Expert`, `CallableExpert`, `BoundExpert`, `linear` / `LinearExpert`,
  `Conv1dExpert`, `Conv2dExpert`, `ConvTranspose1dExpert`, `ParamExpertBank`
- `ExpertFn` is a compatibility alias of `Expert`
- `expand_param`, `bank_weight_shape`; optional `expand_expert` for stock ops only
- `anytrain.module.moe.conv`: `ConvRouter1d` / `ConvRouter2d` (logits only),
  plus partitioner / shape helpers for advanced use

### Expert contract

```python
class Expert(Protocol):
    @property
    def bias_kind(self) -> Literal["none", "additive_channel"]: ...
    def __call__(self, input, weight, bias=None, /) -> Tensor: ...
    def bind(self, weight, bias, *, num_experts, grid) -> BoundExpert: ...
```

`BoundExpert` holds `bank` + `partitioner`. Stock experts implement `bind`;
`CallableExpert(fn, bias_kind="none")` covers Identity-only custom fns.
Bare callables (including raw `F.linear`) are not the public contract;
`F.linear` is coerced to `LinearExpert()` at MoE construction for convenience.

Masked dense or sparse routing may produce a step with no active route. On the
first such step, module-backed and custom generic experts are called with an
empty batch to probe their output shape and dtype; custom experts therefore
need to preserve their normal non-batch output contract for
`input.shape[0] == 0`. The empty probe avoids executing masked token data or
updating stateful layers with it. Dense Linear and generic banks also compact
partially masked inputs and execute only active positions before scattering
outputs back to the original layout.

### Parameter helpers (not Module compilation)

Reliable shared helper:

- `expand_param(single_weight, num_experts, *, init=...)` → `(E, *single.shape)`
- `bank_weight_shape(module_or_weight, num_experts)` → that size

Optional stock-only factory:

- `expand_expert(module, num_experts, *, init=...)` accepts **only**
  `nn.Linear`, `nn.Conv1d` (incl. `moe.conv.Conv1d` for `causal`), `nn.Conv2d`,
  and `nn.ConvTranspose1d`. It copies geometry into the matching `Expert` and
  expands params with `expand_param`. Anything else raises `TypeError`.

It is **not** a general “infer Expert from `nn.Module`” API. Nested modules,
activations, residual blocks, and other custom `nn.Module` graphs cannot be
mapped safely to a single expert call; write the `Expert` and shapes explicitly.

`init="copy"` tiles the template so dense mixtures and sparse one-hot routes
match the template op. `kaiming` / `empty` are independent bank inits.

### Why Conv and ConvTranspose stay separate

Param expansion is already shared (`expand_param` / expert-major `(E, ...)`).
Conv and ConvTranspose do **not** share one expert/runtime path because their
contracts differ:

- weight layout: conv is `(out, in/g, k...)`, transpose is `(in, out/g, k...)`
- geometry knobs: conv has `causal` / `padding_mode`; transpose has `output_padding`
- restore: concat for Conv1d grid cells vs overlap-add for ConvTranspose1d

Bias handling is shared for Conv1d and ConvTranspose1d: cell path is
weight-only; after restore apply
``restore(bias_cells) / restore(coverage_cells)`` (additive channel bias;
coverage is 1 for concat, >1 on transpose overlap-add).

Reuse the param factory; keep `Conv*Expert` / partitioners as distinct families.

Router produces logits. Gate (default softmax, or ADT) produces the full
probability distribution. Sparse combine weights use top-k on logits followed by
softmax over the selected logits — controlled by `top_k`, not by the gate.

`anytrain.module` does not re-export these names at the package top level.

## Grid Contract

- `grid=None` / `NativeGrid`: token MoE, linear experts, or per-sample Conv2d.
  `grid=None` is normalized to `NativeGrid()` before `Expert.bind()`.
- `Grid1d(segment_size, context=None, fill=None)`: tile a 1D axis into routing
  segments. `segment_size` is the logical owned region / routing stride, not
  necessarily the tensor length passed to the expert.
- `context=None` defers to the expert. `context="global"` lets each segment
  materialize the expert's required left/right context from neighboring real
  input positions; `context="local"` treats every segment as an independent
  local boundary.
- `fill=None` defers to the expert. `fill` controls how missing materialized
  positions are filled when global boundaries, local boundaries, or tails need
  extra values. It is grid materialization fill, not a second copy of conv
  `padding`.
- `Conv1dExpert` resolves missing `context` to `"global"` and missing `fill`
  to the expert's `padding_mode`. `ConvTranspose1dExpert` only uses
  `segment_size`; explicit `context` must be `"global"` and explicit `fill`
  must be `"zeros"`.

## Expert Execution

| `expert` | `weight` shape |
|---|---|
| `linear` / `LinearExpert` | `(E, out, in)` |
| `Conv1dExpert` | `(E, out, in/groups, k)` |
| `Conv2dExpert` | `(E, out, in/groups, kH, kW)` |
| `ConvTranspose1dExpert` | `(E, in, out/groups, k)` |
| `CallableExpert` / custom `Expert` | `(E, ...)` with Identity unless `bind` says otherwise |

Dense (`top_k=None`) linear execution runs the full activation batch once per
expert and accumulates weighted outputs; convolution families mix parameters
before their grouped op. Sparse (`top_k=K`) groups routed activations by expert,
runs one batch per used expert, then combines outputs with the top-k softmax
weights. Neither linear path materializes complete expert weights per routed
token. Dense execution keeps only one `[tokens, output]` accumulator in the
forward loop; autograd may still retain each expert output while router-weight
gradients are required, for `O(experts * tokens * output)` saved tensors. Portable
sparse dispatch reads the expert segment counts from the device once per bank
call; it does not synchronize or invoke an expert per token-route pair. Cell
execution is weight-only for `additive_channel`; mixed bias is applied additively
after restore.

The routing mask is enforced at the expert-bank output boundary. Masked cells
have zero mixed expert weight and zero additive bias, so they cannot contribute
through either the weight-only cell path or the post-restore bias path.
`ConvTranspose1dExpert` overlap coverage counts only active cells; masked cells
therefore cannot change the overlap-add normalizer for neighboring active cells.
Native convolution masks follow the batch routing shape. `Grid1d` masks may use
the logical `[batch, num_segments]` cell shape (including broadcastable forms)
or the flattened `[batch * num_segments]` routing shape.

## ADT Integration

ADT is only a gate: `MoE(..., gate=AdaptiveDirichletTempering(...))`. Router and
gate both take `num_experts` explicitly at construction; MoE validates they
match the top-level value. ADT does not live inside the router, select experts,
partition inputs, dispatch experts, or emit Lightning logs.

## Auxiliary Loss and Logging

MoE forward returns `MoEOutput` only. Downstream code composes task loss and
MoE aux losses explicitly. `MoE` registers one routing `OutputObservation` in
the shared `anytrain.observation` registry. `ObservationCallback` discovers the
module from the `pl_module` tree and records effective experts plus scalar
summaries for usage, gate probability, actual route weight, tokens, and dropped
ratio. This is the only MoE-intermediate logging path in anytrain; direct
`.route()` / `.forward_manually()` calls bypass the normal module forward hook.

`moe_loss()` runs after forward and is an explicit training objective, not a
module intermediate. Downstream code records it through normal loss logging
(`self.log` or its own scalar loss composition) rather than injecting it into
the observation registry.

## Limits

- Local PyTorch execution is the default. Expert parallelism, all-to-all,
  capacity-factor dropping, and custom fused kernels are left to downstream
  projects or later targeted additions.
- MoE forward never adds auxiliary losses implicitly.
