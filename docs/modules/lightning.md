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
  schedule/
    __init__.py
  callback/
    __init__.py
    checkpoint.py
    dirichlet_tempering.py
    debug.py
    ema.py
    grad.py
    loss.py
    parameter_policy.py
    performance.py
    quantization.py
    moe.py
  mixin/
    __init__.py
    log.py
```

当前公开导出：

- `experiment`：从 Trainer 主 logger 获取 text/scalar/audio/histogram 能力。
- `validation`：加权 validation logging 与可恢复指标历史。
- `schedule`：统一 step/unit LR schedule spec、unit 进度、unit LR runtime 和停止控制。
- `LightningLogMixin`
- `ModelCheckpoint`
- `DebugCallback`
- `EMACallback`
- `ParameterEMA`
- `find_ema_callback`
- `ParameterPolicyCallback`
- `apply_parameter_trainability`
- `GradientProbeLoggerCallback`
- `GradientProbe`
- `GradientTarget`
- `GradientComparison`
- `LossItemLoggerCallback`
- `LossSummaryCallback`
- `LossTimeBucketLoggerCallback`
- `FlopsProvider`
- `PerformanceCallback`
- `ADTStatsLoggerCallback`
- `ADTStatsMonitor`
- `CodebookUsageLoggerCallback`
- `CodebookUsageMonitor`
- `CodebookUsageMonitorsFn`
- `QuantizationLoggerCallback`
- `QuantizationMonitor`
- `QuantizationMonitorsFn`
- `MoELoggerCallback`
- `MoEMonitor`
- `RankLogMode`
- `prefixed_log_dict`

`anytrain.lightning` 不导出自定义 LightningModule 基类，也不通过继承改变训练行为。下游项目直接继承 `lightning.pytorch.LightningModule`。

## Schedule

`anytrain.lightning.schedule` 提供统一的 LR schedule spec，但按 clock 选择不同执行机制。
下游用同一个 `LRSchedule(phases=...)` 描述 warmup、stable、decay 等阶段；`StepClock`
编译成 Lightning 原生 step scheduler，`UnitClock` 则不返回 Lightning scheduler，而是由
`ScheduleRuntime.callbacks()` 生成唯一 owner 的 unit runtime callback。`Phase.duration` 和
`LRSchedule.milestones()` 的单位都由 clock 决定：step clock 下是 optimizer steps，unit clock
下是 global valid units。

下游仍显式创建 optimizer，并在 `configure_optimizers()` 中调用
`runtime.configure_optimizers(optimizer)`。step clock 返回 optimizer + Lightning scheduler；unit
clock 只返回 optimizer。Trainer callback 列表加入 `*runtime.callbacks()`。unit runtime 在 train
batch end 读取下游 `UnitProvider`、按 rank sum 得到 global valid/padded units、维护 checkpoint
state、只在 optimizer step boundary 更新 LR，并在达到 `stop_at` 或 `stop_at_end` 时设置
`trainer.should_stop=True`。它会拒绝和 Lightning `lr_scheduler` 同时启用，并默认检查 optimizer
LR 没有被其它 callback 改写。日志使用 `progress/*` 命名，例如 `progress/tokens_total`、
`progress/tokens_per_second_window`、`progress/padding_ratio` 和 `progress/lr_ratio`。schedule
runtime 不重建 optimizer、不改变 param group 结构，也不解释 batch schema。

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

`EMACallback` 维护可训练浮点参数的 float32 指数滑动平均。它在 `trainer.global_step` 前进时更新
shadow weights，并在 validation / test / predict 开始时把 EMA 权重 swap 进模块、结束时 restore。
下游也可以通过 `find_ema_callback(trainer)` 与 `average_parameters(pl_module)` 在 sample / artifact
导出等自定义 hook 里临时使用 EMA 权重。callback 把 EMA state 写入 Lightning checkpoint；它不解释
batch schema，也不自动改写 `state_dict` 里的原始训练权重。Lightning 会在 `on_fit_start` 之前恢复
callback state，因此 `load_state_dict` 会暂存 EMA payload，并在 `on_fit_start` 创建 shadow 后再加载。

`ParameterPolicyCallback` 按下游策略设置参数 `requires_grad`。它在 callback `setup(stage="fit")`
应用策略，早于 Lightning 调用 `configure_optimizers()`，避免冻结参数仍进入 optimizer、或解冻参数
未被收集。`on_fit_start` 仍保留为幂等回退，方便单测直接调用。

`ModelCheckpoint` 继承 Lightning 原生 `ModelCheckpoint`。默认 `async_save=True`，rank 0 先保存到本机临时目录，再把复制和删除操作排进单线程后台队列。它用于目标 checkpoint 目录位于 NFS 等慢文件系统时，缩短训练主循环等待目标文件系统写入的时间；传入 `async_save=False` 时保持原版同步行为。

异步模式要求 `trainer.save_checkpoint()` 产出普通文件，并明确不支持 FSDP/DeepSpeed 的目录型或分片 checkpoint；这类训练应设置 `async_save=False`。异步复制完成后才通知 logger 的 `after_save_checkpoint`。

`PerformanceCallback` 记录参数量、optimizer step time、硬件峰值算力和 MFU。固定计算量使用
`model_flops_per_step`；变长 batch 使用 `model_flops_per_batch: FlopsProvider`，callback 在
梯度累积的 optimizer 边界汇总 microbatch FLOPs。动态输入、DDP 聚合和日志口径见
[`lightning.perf` 设计](perf.md)。DDP 默认在 timer 前 barrier 对齐 rank；`sync_distributed=False` 可显式
关闭该行为。

`ADTStatsLoggerCallback` 默认从 `pl_module.named_modules()` 自动发现所有
`AdaptiveDirichletTempering` module，读取它们的 `monitor`，并用通用 monitor naming helper
推断最短唯一日志名：单个 `encoder.router.gate` 记录为 `adt/gate/temperature_mean`；如果同时存在
`encoder.router.gate` 和 `decoder.router.gate`，则分别记录为
`adt/encoder.router.gate/temperature_mean` 和 `adt/decoder.router.gate/temperature_mean`。显式 mapping key 或
`ADTStatsMonitor(name=...)` 会覆盖自动推断；裸 `ADTStats` 无法反查来源时使用稳定 fallback
如 `adt/monitor_0/...`。callback 不解释 batch，也不额外触发 router forward。它把
`adt_stats_metrics()` 生成的 ADT-only scalar summary 加上 `tag/name` 前缀记录，默认只包含
variance、alpha 和 temperature；gate utilization / effective experts 由 gate 或 MoE routing logger
记录，避免和 ADT 重复。完整 ADT per-expert 分布只在 `log_histograms=True` 时作为 histogram 写入。
DDP 下 callback 不做额外 collective。
ADT 默认像 BatchNorm 一样只用本地 stats；跨卡同步若以后需要，会以 SyncBatchNorm 风格的
显式 API 提供，而不是构造参数自动推断。

`CodebookUsageLoggerCallback` 只消费下游显式返回的 `CodebookUsageMonitor` mapping；
monitor 绑定任意离散 `indices`、对应 `codebook_sizes` 和可选
`active_codebook_mask`。它不扫描 `pl_module`，不解释 batch，也不为了日志额外 forward
模型。单进程直接用当前 batch 的 indices 统计 usage；DDP 下先对每个 codebook 的 counts、
active count 和 denominator 做 sum 归约，再计算 perplexity、dead code ratio 和 active ratio，
避免平均各 rank 比例。histogram 默认关闭，开启后只由 global rank 0 写入本地观测到的 index 分布。

`QuantizationLoggerCallback` 只消费下游显式返回的 `QuantizationMonitor` mapping；monitor 绑定
`QuantizeOutput` 和对应 quantizer 或显式 `codebook_sizes`。callback 记录 quantization loss，
并把 `QuantizeOutput.indices` / `active_codebook_mask` 转交给同一套 codebook usage 统计；
它同样不扫描 `pl_module`，不解释 batch，也不为了日志额外 forward quantizer。

`MoELoggerCallback` 默认从 `training_step` outputs mapping 中自动发现 `MoEOutput` /
`ExpertRouting` / `MoEMonitor` values；例如返回 `{"router": moe_output}` 时会记录到
`moe/router/...`。同名 `MoELossOutput` 可用 `router_loss` / `router_aux_loss` /
`router_moe_loss` / `router/loss` 自动配对；需要自定义抽取时再传显式 monitor function。
callback 不解释 batch，也不为了日志额外 forward router 或 expert。它把
`moe_routing_metrics()` 生成的 routing scalar 和 `moe_loss()` 返回值里的 loss scalars 加上
`tag/name` 前缀后记录，例如 `moe/router/effective_experts`、
`moe/router/usage_mean`、`moe/router/gate_probability_mean`、
`moe/router/route_weight_mean`、`moe/router/tokens/total`、`moe/router/dropped/ratio`
和 `moe/router/loss/total`。默认不为每个 expert 生成 scalar 曲线；需要看完整
per-expert 分布时设置 `log_histograms=True`，以 histogram 记录 usage、
gate_probability、route_weight 和 tokens。`gate_probability` 是 gate 输出的 full distribution；
`route_weight` 是 MoE 实际执行时使用的 combine weight，因此 `top_k=None` 使用 dense
probabilities，`top_k=K` 使用 materialized selected weights。loss logs 只在 monitor
提供或 outputs 中可自动配对 `MoELossOutput` 时记录。若 outputs key 与
`module_monitor_names(pl_module, MoE)` 推断出的 shortest suffix 匹配，且对应 `MoE.gate`
是 `AdaptiveDirichletTempering`，callback 会在同一 prefix 下追加 `adt/...` stats；
这些嵌套 ADT stats 默认只包含 ADT 独有的 variance、alpha 和 temperature，usage/effective experts
继续由 MoE routing metrics 记录。
独立非 MoE 的 ADT 仍使用 `ADTStatsLoggerCallback`。DDP 下 callback 不做额外 collective；
需要全局 MoE 统计时，下游应先在任务层聚合 routing outputs 或 metrics。

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
- ADT、quantization 和 MoE callback 的显式 monitor contract、日志 cadence 和 root export。
- callback 可直接传入 Lightning `Trainer`。

后续新增 logger backend 或 callback 时，需要补充与 Lightning logger backend 的集成测试，并确保没有引入 optional 依赖到 core import。
