# Automatic Cost Calibration

anytrain.lightning.auto_cost measures temporary CUDA memory for arbitrary
training batches. It does not inspect a batch type, model attribute, objective
name, or LightningModule implementation.

Projects provide already-collated probe batches and an AutoCostHooks adapter.
The adapter supplies prepare, device transfer, actual cost, and one explicit
forward/backward operation. The engine records successful actual-cost and
peak-memory points plus probe targets skipped after synchronized OOM.

fit_memory_calibration converts successful points into a linear slope and
intercept. When peak memory is slightly superlinear (typical of two CUDA
probes), OLS can yield a negative intercept; the fit then falls back to a
through-origin envelope ``intercept=0`` and ``slope=max(peak/cost)`` so
``resolve_auto_cost`` stays conservative. resolve_auto_cost applies free CUDA
memory, headroom, and minimum cost policy. Dataloader sample capacity and
planning-window policy stay in the project because they depend on the project
planner.

For FSDP, invoke calibration after the strategy has sharded parameters and
pass the wrapped strategy model. The same hooks and distributed OOM behavior
apply; the project remains responsible for replacing its datamodule after the
resolved plan is available.
