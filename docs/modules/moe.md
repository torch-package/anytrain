# anytrain.module.moe Design

## Positioning

`anytrain.module.moe` provides a thin, task-agnostic token-choice MoE component set for downstream LightningModules.

It deliberately stops at local sparse routing mechanics: it does not provide expert registries, model-zoo blocks, training entrypoints, batch-schema assumptions, expert parallel all-to-all, or hidden logging behavior.

## Public API

The subpackage exports:

- `TopKRouter` / `LinearTopKRouter`: feature-last linear router from `(..., input_dim)` hidden states to top-k expert assignments.
- `TokenChoiceMoE` / `MoELayer`: wrapper that runs selected local `nn.Module` experts and combines weighted outputs.
- `anytrain.module.moe.conv`: convolutional dense-mixture MoE layers (`DynamicConv1d`, `DynamicConv2d`, and `DynamicConvTranspose1d`).
- `RouterOutput`: MoE-compatible subclass of `anytrain.module.moe.routing.SparseExpertRouting`, carrying router logits, probabilities, selected expert indices, selected combine weights, and optional token mask.
- `MoEOutput`: combined hidden states, routing output, per-expert assignment counts, and dropped-token mask.
- `dispatch_to_experts()` / `sparse_dispatch()`: standalone sparse module dispatch/combine helper from `anytrain.module.moe.sparse` for manually supplied `SparseExpertRouting` / `RouterOutput`.
- `load_balance_loss()`, `router_z_loss()`, `sparse_moe_aux_loss()`, and `SparseMoELoss`: explicit auxiliary loss helpers for downstream code to weight and add to task objectives.
- `expert_usage()`, `effective_experts()`, `expert_probability_mean()`, `tokens_per_expert()`, `dropped_token_ratio()`, `sparse_moe_routing_metrics()`, and `sparse_moe_loss_metrics()`: pure scalar metric helpers for explicit logging.

`anytrain.module` does not re-export these names at the package top level. Users should import from `anytrain.module.moe` explicitly.

## Tensor Contract

Sparse MoE and MoE conv share only the routing data contract via `anytrain.module.moe.routing`; expert execution remains subdomain-specific.

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

## Auxiliary Loss and Logging

Sparse MoE forward returns `MoEOutput`; it never mutates the training objective or emits logs by itself. Downstream LightningModules should keep task loss composition explicit:

```python
output = self.moe(hidden_states, mask=mask)
task_loss = self.task_loss(output.hidden_states, target)
moe_loss = sparse_moe_aux_loss(output, load_balance_weight=0.01, router_z_weight=0.001)
loss = task_loss + moe_loss.total
```

`SparseMoELoss` provides the same computation as a small `nn.Module` wrapper when users prefer to configure coefficients in `__init__`. By default it returns the scalar aux loss; with `return_details=True`, it returns `(loss, details)` where details contain detached scalar components such as `load_balance`, `router_z`, and their weighted values.

`SparseMoELoggerCallback` lives under `anytrain.lightning`, not inside the MoE layer. It consumes a user-supplied monitor function that returns named `SparseMoEMonitor` objects, so the callback does not scan `pl_module`, interpret batches, or trigger another router forward. A typical callback wiring is:

```python
from anytrain.lightning import SparseMoELoggerCallback, SparseMoEMonitor

callbacks = [
    SparseMoELoggerCallback(
        lambda outputs, batch, pl_module: {
            "router": SparseMoEMonitor(outputs["moe"], loss=outputs.get("moe_aux_loss")),
        },
        every_n_steps=100,
    )
]
```

Default sparse MoE logs are relative scalar keys under `moe/<monitor>/...`, including effective experts, per-expert observed usage, mean router probability, assignment counts, dropped-token ratio, and optional aux-loss components. Loss logs are emitted only when `SparseMoEMonitor(loss=...)` is provided. The callback does not do distributed reduction; if a project needs global router statistics, it should aggregate routing outputs or metrics at the task layer before exposing the monitor.

## First-Version Limits

- Dispatch is local PyTorch dispatch, optimized for clarity and small/medium expert sets rather than fused kernels.
- Experts are caller-supplied modules and must preserve the routed token batch dimension.
- Capacity factors, token dropping beyond mask handling, expert parallelism, and custom kernels are intentionally left to downstream projects or later targeted additions.
- MoE forward never adds auxiliary losses implicitly; downstream LightningModules should combine `sparse_moe_aux_loss()` / `SparseMoELoss` with task losses using explicit coefficients.
