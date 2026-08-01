# `anytrain.module.routing` Design

## Positioning

`anytrain.module.routing` is the neutral expert-routing contract shared by MoE-family components that choose among experts.

It deliberately models routing data and validation only. It does not execute experts, mix kernels, create expert registries, add auxiliary losses, perform distributed all-to-all, or assume a downstream batch schema.

## Public API

The module exports:

- `DenseExpertWeights`: dense mixture weights with expert axis on the last dimension. `anytrain.module.moe.conv` uses this contract for per-sample or per-segment kernel mixtures.
- `SparseExpertRouting`: sparse top-k routing result with `logits`, dense `probabilities`, selected `indices`, selected combine `weights`, and optional boolean `mask`. MoE `RouterOutput` subclasses this contract for backward-compatible imports from `anytrain.module.moe`.
- `validate_num_experts()` and `validate_top_k()`: shared scalar parameter checks.
- `validate_dense_expert_weights()`: validates dense expert weights and returns the underlying tensor.
- `validate_sparse_expert_routing()`: validates sparse route shape, expert ids, dtype, and device.
- `broadcast_routing_mask()` and `flat_active_mask()`: shared boolean mask broadcasting for router, dispatch, and metrics.

`anytrain.module` does not re-export these names at the package top level. Users should import from `anytrain.module.routing` explicitly when they need the shared routing objects.

## Boundary

Shared abstraction stops at the route:

- `anytrain.module.moe.conv` consumes dense expert weights, then mixes convolution kernels and bias tensors.
- Sparse MoE consumes sparse expert routing, then dispatches feature-last tokens to local experts and combines outputs.
- Metrics such as MoE load-balance loss and z-loss remain in `anytrain.module.moe`, because they are sparse-router training objectives rather than generic routing data.
- ADT remains a logits calibration/statistics component; routing modules decide whether to use calibrated probabilities densely or sparsely.

This keeps expert selection reusable without turning `anytrain.module` into a model zoo or hidden execution framework.
