# anytrain.lightning.perf

## 定位

`anytrain.lightning.perf` 提供训练效率观测的通用计算组件。它定义 MFU（Model FLOPs Utilization）的统一口径，并提供参数量统计、代表性 forward FLOPs profiling、硬件峰值算力推断和 Lightning callback。

核心指标：

```text
MFU = model_flops_per_step / step_time / hardware_peak_flops
```

MFU 数值越高越好。`anytrain` 不额外记录反向含义的训练效率指标。

窗口指标不对逐 step MFU 求平均，而是按总工作量和总时间计算：

```text
window_mfu = sum(model_flops) / sum(step_time) / hardware_peak_flops
```

## 边界

`anytrain.lightning.perf` 负责：

- 统计模型参数量。
- 用 PyTorch profiler 在下游提供的代表性输入上估算 forward FLOPs。
- 提供 `training_flops_from_forward()`，显式把 forward FLOPs 转成训练 step FLOPs。
- 根据 GPU 型号和 compute dtype 查内置硬件表，得到硬件峰值算力。
- 在 Lightning 训练中记录 step time 和 MFU。

`anytrain.lightning.perf` 不负责：

- 替下游选择代表性输入。
- 猜测 batch schema。
- 猜测 tokens、frames、valid mask count 等任务数据量。unit 进度、padding 和吞吐日志由
  `anytrain.lightning.schedule` 的 `UnitClock` / `UnitScheduleCallback` 负责。
- 接管 Hydra、Trainer 或训练入口装配。

下游 job 可以显式覆盖 `hardware_peak_flops`。覆盖值会在日志元数据中标记为 `override`。

## 固定 FLOPs

```python
import torch
from anytrain.lightning import PerformanceCallback
from anytrain.lightning.perf import profile_forward_flops, training_flops_from_forward

forward_flops = profile_forward_flops(model, args=(example_batch,))
model_flops_per_step = training_flops_from_forward(forward_flops)

callback = PerformanceCallback(
    model_flops_per_step=model_flops_per_step,
    compute_dtype=torch.bfloat16,
)
```

`model_flops_per_step` 表示当前 rank 每次 optimizer update 的训练 FLOPs。它保留给每步
shape 和计算路径固定的训练；使用梯度累积时，调用方传入的值应已经包含该 optimizer update
内全部 microbatch 的 FLOPs。

如果模型包含 PyTorch profiler 尚不支持 FLOPs 的算子，应由下游直接传入 `model_flops_per_step`。

## 动态 FLOPs

变长 batch 或动态计算路径使用 `model_flops_per_batch`。它接收一个 `FlopsProvider`，callback
在每次 `training_step` 对应的 train batch 结束后调用 provider：

```python
from typing import Any

from lightning import pytorch as pl

from anytrain.lightning import PerformanceCallback


class SequenceFlops:
    def __init__(self, *, flops_per_valid_token: float) -> None:
        self.flops_per_valid_token = flops_per_valid_token

    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> float:
        del trainer, pl_module, outputs, batch_idx
        return float(batch.valid_tokens) * self.flops_per_valid_token


callback = PerformanceCallback(
    model_flops_per_batch=SequenceFlops(flops_per_valid_token=1.2e9),
    hardware_peak_flops=None,
)
```

provider 返回当前 rank、当前 train batch/microbatch 的训练 FLOPs，必须是 finite positive
number。它可以按下游 batch、module 和 `training_step` output 解释动态结构，但应保持轻量、
确定且无 I/O。`model_flops_per_batch` 和 `model_flops_per_step` 互斥。

## 真实 Batch Profiling

当下游还没有解析 batch 的 FLOPs provider 时，可以显式对真实 train batch 启用 PyTorch dispatch
FLOPs counter：

```python
callback = PerformanceCallback(
    profile_flops=True,
    hardware_peak_flops=None,
)
```

counter 从 `on_train_batch_start` 持续到 `on_train_batch_end`，覆盖 Lightning 自动优化中的 forward、
backward 和 optimizer step，并按当前 batch shape 记录动态 FLOPs。`profile_flops` 与
`model_flops_per_step`、`model_flops_per_batch` 互斥；metadata 中记录
`perf/model_flops_source=profile`。

dispatch profiling 会增加 CPU 调度开销，因此它面向 calibration、smoke 和短 screening，不应默认用于
长时间生产训练。需要同时报告无 profiler 的真实吞吐时，先用相同数据顺序做 calibration，再关闭
`profile_flops` 运行独立的 timing job；不能把带 profiler 的 step time 冒充生产吞吐。

## Optimizer 与梯度累积边界

callback 用 `trainer.global_step` 的推进识别 optimizer update 边界：

- 自动优化且无梯度累积时，一个 train batch 对应一个 optimizer update。
- 自动优化且有梯度累积时，provider 仍按 microbatch 返回 FLOPs；callback 把这期间的 FLOPs
  和 batch time 相加，在 optimizer update 完成后形成一个 measurement。
- 静态 `model_flops_per_step` 已经是 optimizer update 口径，不会再按 microbatch 数量放大。
- 手动优化必须在 train batch 内推进 `global_step`。一次 batch 内发生多个 optimizer update 时，
  callback 把自上次边界以来的总 FLOPs/time 按 update 数量报告；它不解释多个 optimizer 的
  独立 FLOPs。

`warmup_steps`、`log_every_n_steps` 和日志中的 step 均使用 optimizer `global_step`。
`measure_window_steps` 保存最近的 optimizer-boundary measurements。尚未推进 `global_step` 的
epoch 末尾 microbatch 不构成完整 measurement，不进入 MFU，也不会带入下一 epoch。

## DDP 边界

DDP 下 provider 和静态值都使用 local-rank 口径，调用方不能预先乘 `world_size`。在每个日志点，
callback 通过 Lightning strategy 聚合窗口：

```text
DDP window MFU = sum_step(sum_rank(FLOPs))
                 / sum_step(max_rank(elapsed))
                 / sum_rank(peak FLOPs)
```

`hardware_peak_flops` 是当前 rank 对应单设备的 FLOP/s；普通同构 GPU 留空自动推断，MIG、
降频或定制设备由 job 按单设备覆盖。`perf/model_flops_per_step` 记录各 rank 的平均 local FLOPs，
`perf/step_time` 使用最慢 rank 的时间，`perf/mfu` 使用上面的全局比值，避免对各 rank MFU
直接求平均。窗口时间先逐 measurement 取最慢 rank，再求和；不能对每个 rank 的窗口总时间
直接取最大值，因为窗口内的最慢 rank 可能随 optimizer update 改变。

默认 `sync_distributed=True`：每个 train batch 的 timer 启动前调用 strategy barrier。这样上一
batch 只在 rank zero 执行的 sample、媒体或 checkpoint callback 不会变成其他 rank 下一步的等待
时间；barrier 发生在 timer 之外。当前 batch 是否包含其他 batch-end callback 的耗时取决于 callback
顺序：要排除诊断，下游应把 `PerformanceCallback` 放在这些 callback 之前。当下游能保证 callback
成本完全对称，或希望测量不含额外 barrier 的连续吞吐时，可显式设为 `False`。

这个聚合契约面向同步 data-parallel，默认一个训练进程对应一个 accelerator。pipeline/model
parallel 的 FLOPs 归属和设备峰值分母需要下游另行定义，callback 不猜测并行拓扑。

## 日志字段

- `perf/model_params`、`perf/model_trainable_params`：当前 module 参数量。
- `perf/model_flops_per_step`：当前 measurement 每 optimizer update、每 rank 的平均 FLOPs。
- `perf/model_flops_per_step_window`：窗口内每 optimizer update、每 rank 的平均 FLOPs。
- `perf/step_time`：当前 measurement 每 optimizer update 的最慢 rank 时间。
- `perf/step_time_window`：窗口内每 optimizer update 的最慢 rank 平均时间。
- `perf/hardware_peak_flops`：当前 rank 的设备理论峰值 FLOP/s。
- `perf/mfu`：窗口 FLOPs 总和除以窗口时间总和及硬件峰值总和。
- `perf/gpu_peak_memory_allocated_bytes`、`perf/gpu_peak_memory_reserved_bytes`：训练开始后 CUDA
  allocator 的峰值；DDP 日志点取各 rank 最大值。
- `perf/model_flops_source`：metadata 字段，取 `static`、`provider`、`profile` 或 `unresolved`。

## Unit 进度边界

有效数据量、padding 和 `{unit}_per_second` 不再属于 `lightning.perf`。当训练需要按
tokens、frames 或 audio seconds 作为可比时间轴时，使用 `anytrain.lightning.schedule`：unit
runtime 只从下游显式 `UnitProvider` 读取 batch schema，并在同一个 owner callback 中完成 DDP
unit 聚合、LR 调度、停止训练、progress 日志和 checkpoint state。`PerformanceCallback` 继续只
负责 FLOPs/MFU、step time 和 memory。
