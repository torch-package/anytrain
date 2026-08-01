# `anytrain.module.moe.routing` Design

## Positioning

`anytrain.module.moe.routing` is the density-neutral expert-routing contract
shared by MoE-family components.

It models routing data and validation only. It does not partition inputs,
execute experts, mix kernels, create expert registries, add auxiliary losses,
perform distributed all-to-all, or assume a downstream batch schema.

Dense mixture (`top_k=None`) and sparse top-k (`top_k=K`) both materialize the
same `ExpertRouting` record.

## Public API

- `ExpertRouting`: density-neutral routing record
- `materialize_routing()` / `expert_routing_from_weights()`
- `validate_num_experts()` / `validate_top_k()` / `validate_expert_routing()`
- `broadcast_routing_mask()` / `flat_active_mask()`
- `DenseExpertWeights` for probability-tensor helpers

## ExpertRouting Contract

```python
@dataclass(eq=False)
class ExpertRouting:
    logits: Tensor                      # (..., num_experts)
    probabilities: Tensor               # (..., num_experts)
    top_k: int | None = None
    indices: Tensor | None = None       # (..., top_k) when top_k is set
    weights: Tensor | None = None       # selected combine weights; also as selected_weights
    mask: Tensor | None = None
```

Routers produce logits. MoE owns gating and density materialization:

```text
logits = router(router_features)
probabilities = gate(logits)                 # full distribution (default softmax)
# top_k=None  -> dense mix with probabilities
# top_k=K     -> indices = topk(logits); weights = softmax(selected_logits)
routing = materialize_routing(logits, probabilities, top_k=runtime_top_k, mask=...)
```

## Boundary

Partition/restore belongs to `MoEPartitioner` in `anytrain.module.moe`. Expert
banks consume `ExpertRouting` and choose an efficient execution path. Metrics
and aux losses remain in `anytrain.module.moe`.
