# anytrain.module.moe Design

## Positioning

`anytrain.module.moe` provides a thin, task-agnostic token-choice MoE component set for downstream LightningModules.

It deliberately stops at local sparse routing mechanics: it does not provide expert registries, model-zoo blocks, training entrypoints, batch-schema assumptions, expert parallel all-to-all, or hidden logging behavior.

## Public API

The subpackage exports:

- `TopKRouter` / `LinearTopKRouter`: feature-last linear router from `(..., input_dim)` hidden states to top-k expert assignments.
- `TokenChoiceMoE` / `MoELayer`: wrapper that runs selected local `nn.Module` experts and combines weighted outputs.
- `anytrain.module.moe.conv`: convolutional dense-mixture MoE layers (`DynamicConv1d`, `DynamicConv2d`, and `DynamicConvTranspose1d`).
- `RouterOutput`: MoE-compatible subclass of `anytrain.module.routing.SparseExpertRouting`, carrying router logits, probabilities, selected expert indices, selected combine weights, and optional token mask.
- `MoEOutput`: combined hidden states, routing output, per-expert assignment counts, and dropped-token mask.
- `dispatch_to_experts()` / `sparse_dispatch()`: standalone dispatch/combine helper for manually supplied `SparseExpertRouting` / `RouterOutput`.
- `load_balance_loss()`, `router_z_loss()`, `expert_usage()`, and `effective_experts()`: pure metric/loss helpers for downstream code to weight and log explicitly.

`anytrain.module` does not re-export these names at the package top level. Users should import from `anytrain.module.moe` explicitly.

## Tensor Contract

Sparse MoE and MoE conv share only the routing data contract via `anytrain.module.routing`; expert execution remains subdomain-specific.

- Hidden states are feature-last: `(..., input_dim)`.
- Router logits and probabilities are shaped `(..., num_experts)`.
- Selected `indices` and `weights` are shaped `(..., top_k)`.
- `mask`, when provided, must be boolean and broadcastable to the leading token shape.
- Masked tokens keep logits/probabilities for diagnostics, but their selected weights are zeroed, they are not dispatched, and they do not contribute to routing metrics.
- `tokens_per_expert` counts active top-k assignments, not unique tokens, so its sum is `active_tokens * top_k` when all active weights are nonzero.
- `TokenChoiceMoE.forward_manually()` accepts any `SparseExpertRouting`; `RouterOutput` remains the public MoE convenience class for backward-compatible imports.

## ADT Integration

`TopKRouter` accepts an optional gate module. Without a gate it applies `softmax` to logits. With `AdaptiveDirichletTempering`, the router passes the broadcast token mask and optional `collect_stats` override through to ADT, then applies top-k selection to the calibrated probabilities.

ADT remains only a logits calibration and utilization-statistics module. Top-k selection, dispatch, auxiliary load-balance loss, z-loss weighting, and Lightning logging stay outside ADT.

## First-Version Limits

- Dispatch is local PyTorch dispatch, optimized for clarity and small/medium expert sets rather than fused kernels.
- Experts are caller-supplied modules and must preserve the routed token batch dimension.
- Capacity factors, token dropping beyond mask handling, expert parallelism, and custom kernels are intentionally left to downstream projects or later targeted additions.
- MoE forward never adds auxiliary losses implicitly; downstream LightningModules should combine `load_balance_loss()` and `router_z_loss()` with explicit coefficients.
