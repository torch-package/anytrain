# `anytrain.lightning` Design

## 定位

`anytrain.lightning` 是 core runtime 层，服务下游普通 LightningModule。训练入口由下游项目自己维护；这个模块只提供任务无关的 logging mixin 和训练调试 callback。

Lightning 是 core 依赖，不作为 optional integration。

## 当前结构

源码结构：

```text
src/anytrain/lightning/
  __init__.py
  experiment.py
  validation.py
  callback/
    __init__.py
    checkpoint.py
    debug.py
    grad.py
    loss.py
    performance.py
  mixin/
    __init__.py
    log.py
```

当前公开导出：

- `experiment`：从 Trainer 主 logger 获取 text/scalar/audio/histogram 能力。
- `validation`：加权 validation logging 与可恢复指标历史。
- `LightningLogMixin`
- `ModelCheckpoint`
- `DebugCallback`
- `GradientLoggerCallback`
- `GradientNormLoggerCallback`
- `LossItemLoggerCallback`
- `LossSummaryCallback`
- `LossTimeBucketLoggerCallback`
- `FlopsProvider`
- `PerformanceCallback`
- `RankLogMode`
- `prefixed_log_dict`

`anytrain.lightning` 不导出自定义 LightningModule 基类，也不通过继承改变训练行为。下游项目直接继承 `lightning.pytorch.LightningModule`。

## Validation

validation 能力从显式子模块使用，不把短名字重新导出到 `anytrain.lightning`：

```python
from anytrain.lightning import validation

validation.log(
    module,
    {"loss": validation.Metric(values, counts)},
)
history = validation.History(prefix="val")
```

`Metric` 是 `anytrain.evaluator.weighted.Metric` 在 validation 子模块中的转发名；它的 weights 是
observation count，而不是任意浮点 importance weight。`log()` 统一负责 epoch/DDP 加权聚合，并在进入
Lightning 的整数 batch-size accumulator 前按 world size 缩放 count，避免不等 rank count 的 mean
reduction 截断分母；`History` 只消费已经归约到 `trainer.callback_metrics` 的 scalar，不解释下游 batch
或指标名。两者共享 `val/` 默认 prefix，但没有对象引用关系；LightningModule 不持有或查找 Trainer
callback。数值归约委托给 `evaluator.weighted.reduce()`，共享 weights tensor 的指标只同步读取一次
count。

下游负责是否创建 validation dataloader、是否执行 validation，以及具体指标按 sample、token 还是 frame
计数。没有 validation dataloader 时 callback 不会自行触发 validation。

## Experiment 能力

`experiment` 统一提供 Trainer 主 logger 的窄能力访问：

```python
from anytrain.lightning import experiment

writer = experiment.audio(trainer)
if writer is not None:
    writer.add_audio("sample", waveform, trainer.global_step, sample_rate=16_000)
```

`text()`、`scalar()`、`audio()` 和 `histogram()` 只在 logger experiment 暴露对应 callable 时返回
类型化 Protocol；没有 logger 或 backend 不支持该能力时返回 `None`。这个模块不选择 logger backend，
也不决定 rank、cadence、tag 或 tensor 的设备转换，这些仍由调用 callback 负责。

## Logging Mixin

`LightningLogMixin` 是无状态 mixin，不持有 LightningModule 或 logger。它在调用时通过当前 module 的 `trainer.loggers` 使用后端能力。

当前方法：

- `log_prefixed_dict(prefix, values, **kwargs)`：给 dict key 加前缀后调用 `LightningModule.log_dict()`。
- `log_audio(tag, audio, sample_rate=..., step=None, rank_mode="zero")`：写入 audio，当前支持 `TensorBoardLogger`。
- `log_figure(tag, figure, step=None, rank_mode="zero")`：写入 figure，当前支持 `TensorBoardLogger`。

如果当前 Trainer 没有 logger，或 logger backend 不支持对应媒体日志能力，方法会直接抛错。

媒体日志默认只在 global rank 0 写入。需要排查分布式 rank 差异时，可以传 `rank_mode="all"`；此时所有 rank 都会写入，且分布式场景下 tag 会自动加上 `rank={global_rank}/` 前缀。

下游项目继续自己实现：

- `forward()`
- `training_step()`
- `validation_step()`
- `test_step()`
- `configure_optimizers()`

## Callback

`DebugCallback` 在调用方显式加入 callback 列表时启用。全部初始参数在 train start 检查一次；
`on_after_backward()` 只重复检查可训练参数与已有梯度。正常路径在设备端聚合 finite 状态，只做少量主机
同步；遇到 NaN 或 Inf 时才定位并打印第一个异常参数或梯度的 name、index、value、shape、dtype、
device，然后直接抛错。正式运行不需要时仍可从 callback 列表移除。

`ModelCheckpoint` 继承 Lightning 原生 `ModelCheckpoint`。默认 `async_save=True`，rank 0 先保存到本机临时目录，再把复制和删除操作排进单线程后台队列。它用于目标 checkpoint 目录位于 NFS 等慢文件系统时，缩短训练主循环等待目标文件系统写入的时间；传入 `async_save=False` 时保持原版同步行为。

异步模式要求 `trainer.save_checkpoint()` 产出普通文件，并明确不支持 FSDP/DeepSpeed 的目录型或分片 checkpoint；这类训练应设置 `async_save=False`。异步复制完成后才通知 logger 的 `after_save_checkpoint`。

`PerformanceCallback` 记录参数量、optimizer step time、硬件峰值算力和 MFU。固定计算量使用
`model_flops_per_step`；变长 batch 使用 `model_flops_per_batch: FlopsProvider`，callback 在
梯度累积的 optimizer 边界汇总 microbatch FLOPs。动态输入、DDP 聚合和日志口径见
[`lightning.perf` 设计](perf.md)。DDP 默认在 timer 前 barrier 对齐 rank；`sync_distributed=False` 可显式
关闭该行为。

`LossSummaryCallback` 在 batch hook 中仅保存 detached device scalar。每个指标的状态固定为前后各
`window_capacity` 个值和完整 observation count；默认 capacity 为 20，不随训练长度增长。训练结束时
所有 rank 先校验 metric schema、window capacity 和 observation count 一致，再共同归约窗口。
`report()` 默认使用配置的 capacity。没有 observation 的空 fit 只在 `on_fit_end()` 确认空报告，不会在
teardown 后发起 collective。Lightning 的 `ddp_spawn` / `ddp_fork` 不会把 worker callback 状态恢复到启动进程，因此这些
launcher 需要在 worker hook 调用父类同步后消费报告；启动进程读取会明确报错。单进程、外部 DDP 和
subprocess DDP 可在 `fit()` 返回后读取。`report()` 在训练结束同步前不可用；`values` 是有界的
host-materialized 兼容视图，不代表完整训练历史。

Python 入口示例：

```python
from lightning import pytorch as pl
from anytrain.lightning import DebugCallback, ModelCheckpoint

callbacks = [
    ModelCheckpoint(dirpath="outputs/checkpoints", async_save=True),
    DebugCallback(),
]

trainer = pl.Trainer(
    callbacks=callbacks,
)
```

## Root 约定

`anytrain.lightning` 不设置 root。下游入口创建 `Trainer` 时显式传入 `default_root_dir`。logger backend 不由 `anytrain.lightning` 自动创建；需要 logger 时，直接通过 Lightning 原生 `trainer.logger` 配置或 Python 代码传入。

第三方 logger backend 属于 optional backend，不放在 core `lightning` 默认导入路径里。

## 边界

`lightning` 不做：

- 不替用户实现任务 step。
- 不解释 batch。
- 不决定 optimizer / scheduler。
- 不内置模型 zoo。
- 不隐式加载 optional domain component。
- 不把 callback registry 做成额外抽象层。
- 不提供 `AnyTrainModule` 这类魔法基类。

## 测试策略

当前覆盖：

- validation count-weighted epoch aggregation、history state 和严格错误路径。
- experiment text/scalar/audio/histogram 能力探测及缺失路径。
- `LightningLogMixin` 的 prefixed dict、媒体 logger 错误路径和 rank logging 策略。
- `DebugCallback` 的异常定位路径和 Trainer 集成。
- `ModelCheckpoint` 的原生接口兼容、异步复制、同步 opt-out 和删除排队。
- callback 可直接传入 Lightning `Trainer`。

后续新增 logger backend 或 callback 时，需要补充与 Lightning logger backend 的集成测试，并确保没有引入 optional 依赖到 core import。
