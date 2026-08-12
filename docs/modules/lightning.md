# `anytrain.lightning` Design

## 定位

`anytrain.lightning` 是 core runtime 层，服务下游普通 LightningModule。训练入口由下游项目自己维护；这个模块只提供任务无关的 logging mixin 和训练调试 callback。

Lightning 是 core 依赖，不作为 optional integration。

## 当前结构

源码结构：

```text
src/anytrain/lightning/
  __init__.py
  _capability.py
  cadence.py
  experiment.py
  materialization.py
  service.py
  oom.py
  validation.py
  schedule/
    __init__.py
  perf/
    __init__.py
  callback/
    __init__.py
    checkpoint.py
    debug.py
    ema.py
    grad.py
    loss.py
    observation.py
    parameter_policy.py
    performance.py
  mixin/
    __init__.py
    log.py
```

当前公开导出：

- `experiment`：从 Trainer 主 logger 获取 text/scalar/audio/image/histogram/raw-histogram 能力。
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
- `GradientLossGroupsFn`
- `gradient_loss_groups`
- `LossSummaryCallback`
- `BackgroundMaterialization`
- `MaterializationCallback`
- `ManagedService`
- `ManagedServiceProvider`
- `ManagedServiceCallback`
- `FlopsProvider`
- `PerformanceCallback`
- `PerformanceReport`
- `ObservationCallback`
- `OOMDiagnostics`
- `oom`：OOM context 标注、输入摘要、错误识别和显式报告 helper
- `RankLogMode`
- `prefixed_log_dict`

callback 的公开 re-export 由 `anytrain.lightning.callback` 唯一维护，root 只从该 package 转发
同一对象。`ObservationCallback` 与其它 callback 一样同时可从 callback package 和 root 导入；
旧的模块专用 observation callback 仍不提供。`tests/test_lightning_api.py` 检查两个入口的
`__all__` 无重复、全部可解析，并防止 root 与 callback package 再次漂移。

`anytrain.lightning` 不导出自定义 LightningModule 基类，也不通过继承改变训练行为。下游项目直接继承 `lightning.pytorch.LightningModule`。

## Schedule

`anytrain.lightning.schedule` 提供统一的 LR schedule spec，但按 clock 选择不同执行机制。
下游用同一个 `LRSchedule(phases=...)` 描述 warmup、stable、decay 等阶段；`StepClock`
编译成 Lightning 原生 step scheduler，`UnitClock` 则不返回 Lightning scheduler，而是由
`ScheduleRuntime.callbacks()` 生成唯一 owner 的 unit runtime callback。`Phase.duration` 和
`LRSchedule.milestones()` 的单位都由 clock 决定：step clock 下是 optimizer steps，unit clock
下是 global valid units。

schedule spec 构造器拥有字段不变量：duration、unit interval、stop position 和 curve ratio 只接受
有限 host real number，不接受 bool、Tensor 或字符串转换；整数窗口严格要求 `int`，phase name 必须
唯一。下游配置层只负责把配置形状转换成这些对象，不重复维护同一组数值校验。

下游仍显式创建 optimizer，并在 `configure_optimizers()` 中调用
`runtime.configure_optimizers(optimizer)`。step clock 返回 optimizer + Lightning scheduler；unit
clock 只返回 optimizer。Trainer callback 列表加入 `*runtime.callbacks()`。unit runtime 在 train
batch end 读取下游 `UnitProvider`，梯度累积期间只在 host 聚合 local valid/padded units。
`UnitBatch.valid` / `padded` 必须是 host real number，不接受 bool 或 Tensor；如果下游只有 device
计数，显式转换及其同步成本由 provider 自己承担。到 optimizer step boundary 才执行一次固定大小
payload 的 rank all-gather 和一次 device-to-host 读取，提交 global unit total、更新 LR，并检查
`stop_at` / `stop_at_end`。DDP throughput 的分子是各 rank unit 总和，分母是该 update 最慢 rank
的 elapsed；窗口则累加每个 update 各自的最慢 rank elapsed。正常 epoch 结束时如果仍有未提交到
optimizer step 的 unit 会直接报错，不会静默丢弃进度。

`UnitClock.sync_cuda` 默认开启，保证正式 throughput key 包含当前 `pl_module` 的 CUDA work；CPU
module 即使系统存在其它可用 CUDA device 也不会触发同步。runtime 在训练开始建立一次计时边界，
之后每个 optimizer step boundary 只同步一次当前 module device，不会在每个 microbatch 首尾同步。
显式设置 `sync_cuda=False` 可以关闭这项计时同步，但异步 GPU work 的实际等待可能落在计时窗口之外，
因此只提供低开销近似值。`measure_window_batches` 继续按
microbatch 数限制窗口，但为保持 optimizer step 原子性只移除完整 update measurement；当一次梯度
累积窗口本身超过该值时，至少保留最新一次 update。日志使用 `progress/*` 命名，例如
`progress/tokens_total`、`progress/tokens_per_second_window`、`progress/padding_ratio` 和
`progress/lr_ratio`。runtime 会拒绝和 Lightning `lr_scheduler` 同时启用，并默认检查 optimizer LR
没有被其它 callback 改写；它不重建 optimizer、不改变 param group 结构，也不解释 batch schema。

schedule 源码 ownership 保持五层：`_curve.py` 定义纯 LR ratio，`_data.py` 定义 unit 输入，
`_spec.py` 定义 immutable schedule，`_runtime.py` 负责 step/unit 编译，`_callback.py` 只负责
unit-clock Lightning lifecycle。callback checkpoint 恢复会先完整校验 totals、optimizer LR group
结构与每个有限非负 host-real LR，再一次性写入实例；不接受字符串/bool 转换，也不会在后续字段
失败时留下部分恢复状态。恢复后的 optimizer 数量必须与保存的 base LR group 数一致。

schedule 验收：

```bash
python -m ruff check src/anytrain/lightning/schedule tests/test_lightning_schedule.py
basedpyright --pythonpath /path/to/python src/anytrain/lightning/schedule \
  tests/test_lightning_schedule.py
python -m pytest -q tests/test_lightning_schedule.py
```

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

`text()`、`scalar()`、`audio()`、`image()`、`histogram()` 和 `histogram_raw()` 只在 logger experiment 暴露对应
callable 时返回类型化 Protocol；没有 logger 或 backend 不支持该能力时返回 `None`。
`histogram_raw()` 用于已经持有 bucket limits/counts 的状态，避免为了渲染浮点 EMA counts 伪造样本。
`image()` 用于已经选成二维并归一化到 `[0, 1]` 的 Heatmap image。
这个模块不选择 logger backend，也不决定 rank、cadence、tag 或 tensor 的设备转换，这些仍由调用
callback 负责。

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

`OOMDiagnostics` 在调用方显式加入 callback 列表时启用。它记录 train/validation batch 当前所处的
Lightning phase，并在未捕获的 Torch OOM 或兼容的 `RuntimeError("out of memory")` 上输出错误、rank、
CUDA memory 和调用上下文。下游通过 `input_reporter` 注入 batch/request 摘要；callback 不解释具体
batch schema，也不重试或拆分 batch。生成、评估等 callback 外调用可以用 `annotate()` 附加调用点上下文，
再由 `report_oom()` 交给已安装的 `OOMDiagnostics`。

`EMACallback` 维护可训练浮点参数的 float32 指数滑动平均。它在 `trainer.global_step` 前进时更新
shadow weights，并在 validation / test / predict 开始时把 EMA 权重 swap 进模块、结束时 restore。
这些 stage 内抛出异常时，callback 会在 `on_exception` 原位恢复训练权重。Lightning 会把
`Checkpoint` callback 排到其它 callback 之后，因此 anytrain 或 Lightning 原生
`ModelCheckpoint(save_on_exception=True)` 会在恢复后保存训练权重，同时 EMA shadow 仍由 callback
state 独立保存；该顺序保证不适用于在 `on_exception` 中自行保存且不继承 Lightning `Checkpoint`
的普通自定义 callback。
下游也可以通过 `find_ema_callback(trainer)` 与 `average_parameters(pl_module)` 在 sample / artifact
导出等自定义 hook 里临时使用 EMA 权重。callback 把 EMA state 写入 Lightning checkpoint；它不解释
batch schema，也不自动改写 `state_dict` 里的原始训练权重。checkpoint state 与 PyTorch module
state dict 一样使用 detached 浅引用；Lightning 在训练继续前同步完成 checkpoint 序列化，因此无需在
每个 distributed rank 上额外 clone 或 D2H 一整套 FP32 shadow。恢复时原位写回已有 shadow。
Lightning 会在 `on_fit_start` 之前恢复
callback state，因此 `load_state_dict` 会暂存 EMA payload，并在 `on_fit_start` 创建 shadow 后再加载。

`ParameterPolicyCallback` 按下游策略设置参数 `requires_grad`。它在 callback `setup(stage="fit")`
应用策略，早于 Lightning 调用 `configure_optimizers()`，避免冻结参数仍进入 optimizer、或解冻参数
未被收集。`on_fit_start` 仍保留为幂等回退，方便单测直接调用。

`GradientProbeLoggerCallback` 要求构造时显式传入 `loss_groups: GradientLossGroupsFn`，不在运行时
隐式猜测 `pl_module` 是否实现特定方法。兼容现有 `current_loss_outputs()` /
`current_gradient_loss_groups()` 的入口是显式 adapter `gradient_loss_groups`；新下游可以直接传入返回
`group -> scalar Tensor mapping` 的项目 provider。callback 校验 provider 的两层 mapping 结构和每个被选
loss 的 0-D Tensor 契约后，再按 `GradientTarget.group` 选择仍连接 autograd 的 loss graph。observation
会 detach 值，因此不能作为 gradient provider。

`GradientProbe`、`GradientTarget`、`GradientComparison` 和 callback 构造器共同维护名称、集合非空、
唯一 probe name、cadence、finite positive `eps` 及 provider callable 等字段不变量；下游配置转换后
无需再次检查这些事实。

`ModelCheckpoint` 继承 Lightning 原生 `ModelCheckpoint`。默认 `async_save=True`，rank 0 先保存到本机临时目录，再把复制和删除操作排进单线程后台队列。队列通过
`max_pending_async_saves` 显式限制持有完整本地 source 的复制任务数，默认值为 1；达到上限时会先
等待最老复制任务、完成对应 callback 或传播错误，再创建下一份完整本地 checkpoint，因此慢 NFS
不会造成临时文件无界积压。删除任务不占该容量，但继续在同一 worker 中保持顺序。DDP 保存前由
global rank 0 完成 backlog/error preflight，并把错误结果同步到所有 rank，避免其它 rank 进入
`trainer.save_checkpoint()` collective 后挂起；fit-end 和 exception drain 也同步传播 rank 0 的后台
错误。`save_last="link"` 只在持久 target 发布成功后原子替换 `last.ckpt`，不链接会被清理的
临时 source；后台复制期间已有 last 保持可读，首次保存则暂不创建。显式 wait 或训练结束 drain
后保证新 last 可读。旧 top-k 仅在该更新成功后删除；更新失败时旧链接及其目标都会保留。它用于目标 checkpoint 目录位于 NFS 等慢文件系统时，缩短训练主循环等待目标文件系统写入的时间；传入 `async_save=False` 时保持原版同步行为。

同一次 top-k 和 `save_last` 保存会复用一份本地序列化结果：先异步发布持久 top-k checkpoint，再从它原子更新或链接 `last.ckpt`，不会重复调用 `trainer.save_checkpoint()`。异常 checkpoint 同样复用该路径；`save_top_k=0` 只保存 `last.ckpt` 一次。

异步模式要求 `trainer.save_checkpoint()` 产出普通文件，并明确不支持 FSDP/DeepSpeed 的目录型或分片 checkpoint；这类训练应设置 `async_save=False`。异步复制完成后才通知 logger 的 `after_save_checkpoint`。

`PerformanceCallback` 记录参数量、optimizer step time、硬件峰值算力和 MFU。固定计算量使用
`model_flops_per_step`；变长 batch 使用 `model_flops_per_batch: FlopsProvider`，callback 在
梯度累积的 optimizer 边界汇总 microbatch FLOPs。动态输入、DDP 聚合和日志口径见
[`lightning.perf` 设计](perf.md)。DDP 默认在 timer 前 barrier 对齐 rank；`sync_distributed=False` 可显式
关闭该行为。

callback 构造器严格校验 cadence/window 的整数边界、FLOPs 的有限正 host-real 契约、互斥 FLOPs
来源及所有 boolean flags；项目配置层应把构造器作为这些字段的唯一运行时 owner。

`ManagedServiceCallback` 通过下游 `ManagedServiceProvider` 获取可选的 task-agnostic 长驻服务。
provider 的 service presence 和 discovery error 在所有 rank 同步；global rank 执行 start、按
optimizer step cadence 执行 health check，并在训练正常结束时 close。owner lifecycle error 会广播
给全部 rank，exception 路径则 best-effort close，不覆盖原始异常。默认 one-shot SIGTERM guard 防止
重复进程组信号重入 Lightning distributed handler。callback 不解释服务的进程实现、设备、数据或
checkpoint/cursor 状态。

`ObservationCallback` 适配 [`anytrain.observation`](observation.md) 的 module 声明。它在
`on_fit_start` 扫描完整 `pl_module.named_modules()`，按 concrete module type 查询唯一 class registry，
claim 所有已发现 declaration，
再按 `covers_descendants` 去除父子重复 observation。默认只有 global rank 0 扫描并注册；显式
`sync_distributed=True` 时才在所有 rank 注册。`OutputObservation` 在普通 `module(...)` 返回后抽取值，
`StateObservation` 在 optimizer step 提交边界读取 module 已维护状态，`ForwardEvent` 则消费 forward 内
显式 `observation.emit()` 的局部值。复制一个正在被观察的 module 时，
复制体上的 runtime forward hook 会在首次调用时自行移除，不能反向写入原 callback。
`observation.context(...)` 使用动态调用上下文给父子 module 的所有 observation 增加公共前缀；train 按
optimizer step 提交，validation 按 validation epoch 独立提交并默认使用 `val` 前缀，两者不共享 pending
state 或 EMA clock。

`BackwardObservation` 当前是 TODO 占位。callback 发现后只发出一次
`BackwardObservationWarning` 并跳过，不注册 backward hook、不替换 output，也不改变 autograd graph。
需要观察 loss 对参数的梯度时使用 `GradientProbeLoggerCallback`；真正的 module backward observation
需要先确定严格只读的契约后再实现。

`callback/observation.py` 当前保持一个内聚的私有 state machine：value delta、step/EMA state、
checkpoint schema、distributed preflight 和 callback lifecycle 共享同一组 private 类型与不变量。
在没有形成独立公开契约前不按文件长度强拆，避免把这些 private state 变成跨模块接口。

运行状态按值类型分开：

- `OutputObservation` / `ForwardEvent` 注册时显式声明 reduction：`Last` 只支持 Curve/Heatmap，`Mean`
  只支持 Curve，`Merge` 只支持 Distribution/Profile/Bins。train 提交窗口是一个 optimizer step，validation
  提交窗口是一个 epoch；它不是 render interval。`Mean` 按 emission 等权，不推断 batch/count。
- `Curve` 按 `Last` 保留窗口最后一次，或按 `Mean` 累加 sum/count；不维护 EMA，也不进入 callback
  checkpoint。没有 history 的 State/Output/Event collector 只在即将 render 的 optimizer step 执行，
  cadence 之外不计算派生 Curve。`StateObservation` 每个边界只 poll 一次，自然按 snapshot 处理。
- `Distribution` / `Bins` 在 callback 消费时立即转成固定 histogram state；`Profile` 把等长的一维 value /
  coordinate pair 转成每个 coordinate bin 的 count、sum 和 sum-squares。它们都不保存完整 output 或
  autograd graph；train 每次 `global_step` 前进时提交 step snapshot，validation 每个 epoch 提交自己的
  snapshot，并分别按各自时钟更新 EMA。Output/Event 必须声明 `Merge + EMAHistory`。
- `Heatmap` 只接受 module 已选好的 2-D floating view，不维护 EMA，且所属 spec 必须声明 `Last`、
  `history=None`；同一提交窗口同一 key 保留最后一次，并只由 global rank 0 转成独立 CPU image。shape/dtype/range
  跨 step 必须稳定。
- `every_n_steps` 控制 train render 和 train 无 history collector，默认每 100 个 optimizer step 输出一次；
  train cadence 外的 State/Output collector 不调用，ForwardEvent payload 直接丢弃。`None` 关闭 train render，
  validation 仍按 epoch 归约。需要 EMA history 的 Distribution/Profile/Bins 仍在对应 stage 的每个提交边界更新，
  非法 finite/range 值也会立即失败。
- spec 的 `EMAHistory.half_life_steps` 是默认值；callback 构造参数 `half_life_steps` 非 `None` 时统一
  覆盖。

callback 在每次 render 的同一批 scalar 中加入 `module/observation_time`。该值累计 interval 内 collector、
state/EMA、distributed reduction、校验、histogram writer 和 Heatmap image writer 的 host wall time，再按
实际 optimizer step 数取平均；它不额外调用 CUDA synchronize，也不包含最终 `pl_module.log_dict()` submission。计时窗口
不进入 checkpoint，`sync_distributed=True` 时保持各 rank 本地口径。

histogram render 使用 `experiment.histogram_raw()` 写 step/EMA bucket counts；backend 不支持 raw
histogram 时仍通过 `pl_module.log_dict()` 写 observations、active ratio、entropy 和 perplexity；
`Bins` 额外写 occupied/dead count 与 ratio。
`max_dense_bins` 在 `_bins_delta` 分配 `[groups, max_bins]` 前检查实际 dense state 大小，防止 flat
mixed-radix codebook OOM。
Heatmap 通过 `experiment.image()` 写出；实际需要 render 但 backend 缺少 `add_image()` 时明确报错。
distributed 模式在 schema preflight 后同步 materialization/backend 错误；`max_heatmap_elements` 超限时要求
module 选择更小的语义 view，不自动 resize 或 sample。

默认 `sync_distributed=False`，只在 global rank 0 保留 rank-local 诊断。开启后每个 optimizer step 先用固定 signature
all-gather 确认所有 rank 的累计 observation keys/schema 和本 step emitted key 集合一致，再 sum
sufficient stats，最后从全局 counts 计算派生指标；signature 同时包含稳定的
`module_path/spec_name/item` source identity 和 reduction，不会把不同 source 或 reducer 恰好拼成的同名
key 归约在一起。`Last` Curve 先保留各 rank 的本地最后一次，再对 rank-local last 求均值。派生 ratio
不跨 rank 求平均。条件执行 module、动态 mapping key 和动态 schema 必须保持 rank 对齐，否则在
进入逐 state reduce 前显式失败。

callback 同时校验内部 state key 和所有实际输出 logger tag 的全局唯一性。检查范围包含 Curve tag、
histogram 的 step/EMA scalar 以及 raw histogram tag，因此 root/child path 或 spec/item 分段方式不同但
最终字符串相同的声明会明确报错，不会在 metric mapping 合并时静默覆盖。

checkpoint state 只包含已完成 step 的 histogram EMA、invalid history、稳定 source identity 和
`last_global_step`，不包含 pending microbatch、Curve、Heatmap 或 runtime binding。source identity 直接绑定当前 module path、
spec name 和 mapping item，允许合法的层级 spec/item 名称而不做 prefix 猜测。attach 留在
`on_fit_start`；loop state 已恢复后，`on_train_start` 再校验 callback step，兼容 Lightning 的 checkpoint
restore 顺序。callback `state_key` 只用稳定的 `tag` 区分实例；恢复要求 `tag` 能匹配旧 callback，并校验
`half_life_steps` / `sync_distributed`。
`every_n_steps` 可在 resume 时改变；`max_dense_bins` 只要仍容纳 checkpoint schema 也可改变。

提交边界通过 train batch end 的 `global_step` delta 判断，因此覆盖自动优化和梯度累积；同一 train
batch 内多个 optimizer step 无法把 pending observation 无歧义拆分，会明确报错。显式 `.forward()` 会绕过
`OutputObservation` hook，但 forward body 内的 `emit()` 和 step boundary 的 `StateObservation` 仍可工作。
不再保留 ADT、quantization/codebook usage 或 MoE 专用 callback/monitor 作为兜底。
量化 loss 是 quantizer forward output 的一部分，可以由其 registry collector 记录；`moe_loss()` 在
forward 外显式计算，属于 task objective，继续走普通 loss logging。

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
- 不把 Trainer callback 做成 registry 或自动装配层；module observation 的 type registry 属于
  `anytrain.observation` 的声明协议。
- 不提供 `AnyTrainModule` 这类魔法基类。

## 测试策略

当前覆盖：

- validation count-weighted epoch aggregation、history state 和严格错误路径。
- experiment text/scalar/audio/histogram/raw-histogram 能力探测及缺失路径。
- `LightningLogMixin` 的 prefixed dict、媒体 logger 错误路径和 rank logging 策略。
- `DebugCallback` 的异常定位路径和 Trainer 集成。
- `ModelCheckpoint` 的原生接口兼容、异步复制、有界背压、同步 opt-out 和删除排队。
- module observation registry 的 exact-type/duplicate/bound-collector 约束，以及 ADT、MoE 和六个
  quantizer 的自动 discovery。
- module observation 的类型约束、State/Output/Event binding、render-only Curve/Heatmap、interval `observation_time`、
  backward TODO warning、rank-zero 默认绑定、梯度累积、step/EMA、DDP schema preflight、dense-bin
  guard、deepcopy/cleanup 和真实 Trainer checkpoint resume。
- callback 可直接传入 Lightning `Trainer`。

后续新增 logger backend 或 callback 时，需要补充与 Lightning logger backend 的集成测试，并确保没有引入 optional 依赖到 core import。

本体模块验收不含另行记录的 schedule/perf：

```bash
python -m ruff check src/anytrain/lightning tests/test_lightning_api.py
basedpyright --pythonpath /path/to/python src/anytrain/lightning \
  tests/test_lightning_api.py
python -m pytest -q tests/test_lightning_api.py tests/test_lightning.py \
  tests/test_lightning_capability.py tests/test_lightning_experiment.py \
  tests/test_lightning_oom.py tests/test_lightning_materialization.py \
  tests/test_lightning_service.py tests/test_lightning_ema.py \
  tests/test_lightning_parameter_policy.py tests/test_lightning_validation.py \
  tests/test_lightning_cadence.py tests/test_lightning_observation.py
```
