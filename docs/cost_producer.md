# Background Cost Producer

anytrain.cost.BackgroundCostProducer is a rank-local bounded worker pool around a
callable that computes and persists one indexed cost. It deliberately has no
dataset dependency: anydataset owns sample identity and manifest writes, while
anytrain owns queue backpressure, lifecycle, and failure propagation.

Submit is non-blocking and returns false when the normal queue is full.
`urgent=True` jumps ahead of that backlog (and bypasses the normal bound) so a
planner miss is not stuck behind speculative prefetch. `num_workers>1` drains
urgent and normal work in parallel so one in-flight `cost_row` cannot serialize
every subsequent wait. The planner's synchronous miss remains the correctness
path. Start the producer only after DataLoader workers are created when the
loader uses forked workers; the S2S loader does this lazily from the planner's
bounded rank-local lookahead.
