# Flow Matching

`anytrain.framework.flow_matching` 已实现为 Facebook `flow_matching` 的可组合薄封装。
安装 optional 依赖后使用：

```bash
python -m pip install flow_matching
```

当前公开提供 source、time sampler、continuous/discrete runtime、objective、sampler 和
Flow-GRPO stochastic transition/log-prob helper；详细
设计与调用示例见
[`docs/flow-matching-design.md`](../../../../docs/flow-matching-design.md)。该模块不接管
下游 `LightningModule`、batch schema 或任务级 reduction。

连续 runtime 默认使用 `UniformTimeSampler(t_min=0.0, t_max=1.0)`；离散 runtime 默认使用 `UniformTimeSampler(t_min=0.0, t_max=1.0 - 1e-3)`，和离散 Euler sampler 的端点保持一致。`LogitNormalTimeSampler` 保留为显式实验选项。离散输入必须是 `torch.long` token tensor。

`ContinuousFlowRuntime` 和 `DiscreteFlowRuntime` 使用同一套 API，持有 path、source、训练时间采样器、generation sampler 和 model caller。objective 只接收对应 runtime；下游需要自己的 mask 和 reduction 时可直接使用 `runtime.training_sample()`，或把 `masked_mse_velocity_loss` 传给 continuous objective。

统一入口为 `objective(model, target, ...)` 和 `runtime.sample(model, source, ...)`。

`anytrain.stats.time_bucketed_mean(values, time, bucket_count=...)` 提供按训练时间分桶的 tensor 聚合，
返回每桶 `total/count/mean`；DDP callback 应同步 `total` 和 `count` 后再计算 mean。
