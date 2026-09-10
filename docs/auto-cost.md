# Automatic Cost Calibration

anytrain.lightning.auto_cost measures temporary CUDA memory for arbitrary
training batches. It does not inspect a batch type, model attribute, objective
name, or LightningModule implementation.

Projects provide already-collated probe batches and an AutoCostHooks adapter.
The adapter supplies prepare, device transfer, actual cost, and one explicit
forward/backward operation. The engine records successful actual-cost and
peak-memory points plus probe targets skipped after synchronized OOM.

fit_memory_calibration converts successful points into a linear slope and
intercept. A negative least-squares intercept caused by measurement noise is
floored at zero; retaining the slope makes the resulting positive-cost estimate
conservative. resolve_auto_cost applies free CUDA memory, headroom, and minimum
cost policy. Dataloader sample capacity and planning-window policy stay in the
project because they depend on the project planner.

Calibration clears the CUDA allocator once before each distinct probe.
Repeated warmup and measurement steps for that probe reuse the allocator;
synchronized baseline and peak-stat resets still isolate each measurement. An
OOM still triggers cache cleanup before the next probe, so this optimization
changes allocator overhead rather than the memory-fit semantics.

For FSDP, invoke calibration after the strategy has sharded parameters and
pass the wrapped strategy model. The same hooks and distributed OOM behavior
apply; the project remains responsible for replacing its datamodule after the
resolved plan is available.
