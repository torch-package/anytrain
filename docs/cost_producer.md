# Background Cost Producer

anytrain.cost.BackgroundCostProducer is a rank-local bounded thread around a
callable that computes and persists one indexed cost. It deliberately has no
dataset dependency: anydataset owns sample identity and manifest writes, while
anytrain owns queue backpressure, lifecycle, and failure propagation.

Submit is non-blocking and returns false when the queue is full. The planner's
synchronous miss remains the correctness path. Start the producer only after
DataLoader workers are created when the loader uses forked workers; the S2S
loader does this lazily from its first indexed-cost prefetch.
