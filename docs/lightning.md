# Lightning

## 定位

`anytrain.lightning` 是面向普通 LightningModule 的核心运行时层。训练入口由下游项目自己维护；这个子包只提供 task-agnostic 的 runtime helper、logging mixin 和调试 callback。

这个包就是服务 Lightning 的，所以 `torch` 和 `lightning` 属于默认测试环境，不再把 Lightning 当成可跳过依赖。

它负责：

- LightningModule 侧的轻量 logging helper。
- Trainer 主 logger 的 text/scalar/audio/image/histogram experiment 能力访问。
- validation 指标按实际 observation count 做 epoch/DDP 加权聚合，并记录可恢复的指标历史。
- 训练 outputs 里的显式命名 scalar loss 窗口汇总 callback。
- 基于显式 live scalar loss provider 的梯度关系和全局梯度范数观测 callback。
- 自动发现普通 module registry 声明的 Curve/Distribution/Profile/Bins/Heatmap observation，并记录
  optimizer-step、EMA 和 render-only image 状态；这是 module 中间量 logging 的唯一 callback。
- non-finite loss 检查等训练调试 callback。
- OOM phase/context/CUDA diagnostics callback；输入摘要由下游 `input_reporter` 提供，不负责重试或拆 batch。
- NFS 等慢文件系统场景下可选的异步 checkpoint 落盘 callback。
- global rank 所有的 task-agnostic 长驻服务启动、健康检查、错误同步和清理 callback。
- 训练效率观测 callback，记录参数量、step time、硬件峰值算力和 MFU。

它不负责：

- 替用户实现具体 `training_step`、`validation_step`、`test_step`。
- 替用户决定 batch schema。
- 替用户决定 optimizer / scheduler。
- 内置模型 zoo、组件加载协议或单模型快捷封装。

下游项目定义自己的 `pl_module`，直接继承 Lightning 原生 `LightningModule`，并按需组合 `anytrain` 提供的 loss、evaluator 或 framework 组件。

## Core 与 Optional

`anytrain` 可以提供训练组件，但按依赖层级拆分：

- core：`observation`、`lightning`、`lightning.schedule`、`lightning.perf`、通用
  loss/evaluator/optim/module 接口和 `module.idspace`。
- optional general：通用 torchmetrics evaluator、第三方 logger backend。
- optional domain：audio/text/speech 等领域 loss、evaluator。
- optional framework：flow matching、MAE、GAN adversarial training 等研究框架。

core import 只依赖默认依赖；optional 子模块可以要求额外依赖，但错误信息应明确提示安装对应 extra。

## 接口原则

- 用户创建自己的 `pl_module`、`data_module` 和 `trainer`。
- 下游入口负责装配和启动训练；LightningModule 负责训练语义。
- 模型和其他子组件作为下游 `pl_module` 的普通字段，由下游类的 `__init__` 明确接收。
- `anytrain` 提供可选训练积木，但不接管组件注入。
- optimizer / scheduler 由下游 `configure_optimizers()` 创建；配置可以作为 `pl_module` 的显式参数传入，也可以直接在代码里手写。
- logger backend 由下游或 Lightning 原生机制配置；`anytrain.lightning` 只提供不持有状态的 logging mixin。

## 类层次

`anytrain.lightning` 不提供自己的 LightningModule 基类，也不通过继承改变训练行为。下游模块直接继承 Lightning：

```python
from lightning import pytorch as pl


class MyPLModule(pl.LightningModule):
    ...
```

会改变训练行为或文件系统状态的能力通过显式 callback、helper 或组件暴露；non-finite loss 检查等调试规则通过 callback 配置。

## Background materialization

`BackgroundMaterialization` 提供 task-agnostic 的 publish/load 生命周期：global owner 在 spawn
进程中调用 producer，发布完成后每个训练进程重新 `load()` 已发布资产。producer、request id 和
产物类型由下游定义；anytrain 不解释 dataset、sample schema、manifest 或 shard。

```python
from anytrain.lightning import BackgroundMaterialization

job = BackgroundMaterialization(
    request_id="filtered-codec-v1",
    producer=producer,
)
job.start(owner=trainer.is_global_zero)
# epoch boundary: owner waits for publication
if trainer.is_global_zero:
    job.finish(owner=True)
trainer.strategy.barrier("filtered_codec_published")
# every rank records that the shared asset is now visible
job.finish(owner=False)
dataset = job.load_ready()
```

worker 默认 `daemon=False`，因为 materializer 往往还要启动 DataLoader、writer 或设备 worker
子进程。只有能够保证 producer 不再 spawn 子进程时才显式使用 `daemon=True`。
`load_ready()` 只允许在同步发布完成后调用；跨 rank 直接使用时，调用方必须按上例在 owner
完成后执行 barrier，再在各 rank 调用 `finish(owner=False)`。训练异常或提前结束时应调用
`close()`；使用 `MaterializationCallback` 时由 callback 负责失败路径回收。

`MaterializationCallback` 通过下游提供的窄 `Materialization` controller 接入 Lightning。它在
train start 同步各 rank 的 active 状态并启动 owner worker，在 epoch end 执行 owner finish、
错误广播、barrier 和 all-rank refresh。业务 datamodule 的方法名和 fallback/schema 仍由下游薄
适配层拥有。

配置形状和运行目录属于下游入口约定，不能藏进 `LightningModule` 继承链里。

## Managed services

`ManagedServiceCallback` 管理与训练并行存在的 task-agnostic 长驻服务。它不创建进程、解释数据、
规划设备或持有训练 cursor；下游只提供一个窄 controller：

```python
from anytrain.lightning import ManagedService, ManagedServiceCallback


def service(trainer) -> ManagedService | None:
    return controller if enabled else None


callback = ManagedServiceCallback(
    service,
    check_every_n_steps=1,
    label="data producer",
)
```

provider 在每个 rank 的 `on_fit_start` 解析，所有 rank 必须一致地返回 service 或 `None`。实际
`start(owner=True)`、`check(owner=True)` 和 `close(owner=True)` 只由 global rank 调用；owner
错误广播给所有 rank。健康检查使用 optimizer `global_step` cadence，同一 step 最多调用一次。
正常 train/fit end 的 close 错误会显式传播；exception 路径只做 best-effort close，避免覆盖原始训练
异常。

默认 `guard_sigterm=True`。callback 在 Lightning 注册 SIGTERM handler 后用 one-shot guard 包装它，
只转发第一次 SIGTERM，避免重复的进程组信号重入 distributed handler。已有其它明确的信号所有者时
可以显式关闭该 guard。服务内部的 subprocess、状态文件、业务阶段和停止协议仍由下游拥有。

## 下游用法

用户仍然手写训练逻辑，但可以直接使用 `anytrain` 提供的组件：

```python
import torch
from lightning import pytorch as pl

from anytrain.evaluator import EvaluatorGroup
from anytrain.loss import LossABC


class MyPLModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: LossABC,
        evaluator: EvaluatorGroup | None = None,
        *,
        lr: float = 0.0003,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.evaluator is None:
            return
        x, y = batch
        y_hat = self(x)
        metrics = self.evaluator(y_hat, y)
        self.log_dict({f"val/metric/{key}": value for key, value in metrics.items()}, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
```

如果下游使用 YAML/JSON 配置，推荐仍然围绕 `pl_module` 组织训练语义，不要叫 `task`：

```yaml
pl_module:
  _target_: my_project.pl_modules.MyPLModule
  model:
    _target_: my_project.models.MyModel
    hidden_dim: 256
  loss_fn:
    _target_: anytrain.loss.LossGroup
    losses:
      reconstruction:
        _target_: torch.nn.L1Loss
  optimizer:
    _target_: torch.optim.AdamW
    _partial_: true
    lr: 0.0003
  evaluator:
    _target_: anytrain.evaluator.EvaluatorGroup
    evaluators:
      error:
        _target_: my_project.evaluator.BatchError
  lr: 0.0003
```

上例的 `BatchError` 是下游实现的无状态 evaluator，因此 validation step 直接调用
`evaluator(...)` 并交给 Lightning 聚合。corpus text metric 则使用独立的 stateful 生命周期：

```python
def validation_step(self, batch, batch_idx):
    prediction_text, target_text = batch
    self.text_evaluator.update(prediction_text, target_text)

def on_validation_epoch_end(self):
    metrics = self.text_evaluator.compute()
    self.text_evaluator.reset()
    self.log_dict({f"val/text/{key}": value for key, value in metrics.items()})
```

`TextComparisonEvaluator.compute()` 在 distributed 已初始化时自行聚合所有 rank 的文本，不能
再把逐 batch BLEU/WER/chrF 交给 Lightning 求均值。只实现 `evaluate()` 的无状态 evaluator
调用 `update/compute/reset` 时会明确抛出 `NotImplementedError`。

如果下游使用领域组件，也还是挂在 `pl_module` 里：

```yaml
pl_module:
  _target_: my_project.audio.AudioPLModule
  loss_fn:
    _target_: anytrain.loss.spectral.MultiScaleSTFTLoss
  evaluator:
    _target_: anytrain.evaluator.speech.SpeechEvaluator
```

这些领域组件属于 optional dependency，不应影响 core import。

## Validation

`anytrain.lightning.validation` 把 validation 的数值聚合与 Lightning 生命周期放在同一个公开功能模块中，
但不要求 LightningModule 持有 callback。下游仍负责计算指标、命名指标以及选择 sample、token、frame
或其他 observation count：

```python
from anytrain.lightning import validation


def validation_step(self, batch, batch_idx):
    values, token_counts = self.validation_values(batch)
    validation.log(
        self,
        {
            "cross_entropy": validation.Metric(values, token_counts),
        },
    )
```

`validation.Metric` 转发 `anytrain.evaluator.weighted.Metric` 的纯数值契约：values/weights shape
必须一致；weights 是有限、非负、整数 observation count，且总数为正。`validation.log()` 固定使用
`on_step=False`、`on_epoch=True`、`sync_dist=True`，并把 count 交给 Lightning 做跨 batch/rank 的
加权均值。内部会按 world size 缩放传给 Lightning 的整数 count，避免 rank 总 count 不能整除 world
size 时被 mean reduction 截断；该共同缩放不改变加权均值。它不接受任意浮点 importance weight，
也不解释指标属于哪个任务。多个指标共享同一 weights tensor 时，纯 `evaluator.weighted.reduce()`
只同步读取一次 count。

需要把 `val/*` epoch 指标写入实验报告时，显式挂载 history callback：

```python
history = validation.History()
trainer = pl.Trainer(callbacks=[history])
trainer.fit(module, datamodule=data)
report = history.report()
```

`History` 区分 sanity 与正式 interval validation；同 step 的重复记录覆盖，state 随 checkpoint 保存和恢复。
validation 已运行但没有匹配 prefix 的 scalar metric 时默认报错；`strict=False` 时跳过该次记录。没有
validation dataloader 时 Lightning 不触发 validation hook，是否启用 validation 仍完全由下游入口决定。

## Experiment 能力访问

callback 需要直接使用 TensorBoard 等 logger experiment 时，通过统一的窄能力接口获取：

```python
from anytrain.lightning import experiment

writer = experiment.scalar(trainer)
if writer is not None:
    writer.add_scalar("train/example", value, trainer.global_step)
```

`experiment.text()`、`scalar()`、`audio()`、`image()`、`histogram()` 和 `histogram_raw()` 检查 Trainer 主 logger
的 experiment 是否暴露对应 callable。`histogram_raw()` 面向已经聚合好的 bucket counts，例如
Distribution/Bins 的 EMA histogram。没有 logger 或 backend 不支持时返回 `None`；helper 不决定 rank、
记录 cadence、tag 或 tensor 设备，下游 callback 继续拥有这些语义。`image()` 面向已经选成二维并归一化的
image tensor。

## Module observations

普通 `nn.Module` 可以通过 `anytrain.observation` 声明推荐观测值，再由一个 Lightning callback 从
完整 `pl_module` 树自动发现和绑定：

```python
from lightning import pytorch as pl

from anytrain.lightning import ObservationCallback


trainer = pl.Trainer(
    callbacks=[
        ObservationCallback(
            every_n_steps=100,
            half_life_steps=1_000,
        )
    ]
)
```

同一个 module 同时用于 train/validation 或同一步存在多个语义调用时，用 observation context 给整棵调用树
增加公共前缀：

```python
from anytrain import observation

with observation.context("reconstruction"):
    loss = self.reconstruction_loss(prediction, target)
```

父 loss、child term 和 forward event 会共享前缀；callback 根据当前 stage 记录为
`train/reconstruction/...` 或 `val/reconstruction/...`。validation observation 在 validation epoch 结束时
独立归约；没有显式 context 时使用 `val/...`。context 不写入 module state。

值类型和历史语义固定为：

- `Curve` 接受 scalar Tensor；`OutputObservation` / `ForwardEvent` 注册时必须显式选择
  `Reduction.Last` 或 `Reduction.Mean`，分别表示保留提交窗口最后一次或按 emission 等权平均。它只记录
  曲线，不维护跨窗口 EMA。
- `Distribution` 接受 1-D floating Tensor 和固定 edges；多维连续 Tensor 必须由 module 先选择明确
  view，不能隐式 flatten。
- `Profile` 接受等长的 1-D floating values/coordinates 和固定 coordinate edges，记录各 interval 的
  value count/mean/std；适合观察 loss 随 timestep、长度等连续坐标的变化。
- `Bins` 接受任意 rank 的离散 Tensor。多 codebook 时用 `group_dim` 指出 codebook 轴，bin 数可以
  大于 1，也可以在不同 group 间不同。
- `Heatmap` 接受 module 已明确选择 sample/head/crop 后的非空 2-D floating Tensor，不维护 EMA，所属 spec
  必须使用 `Reduction.Last` 且 `history=None`；固定 `value_range` 时不同 step 的颜色可比较。同一提交窗口
  同一 key 多次产生时保留最后一次。
- `Distribution` / `Profile` / `Bins` 每个 optimizer step 都先形成固定大小的 sufficient stats，再维护
  EMA；派生指标从 step/EMA state 现算。`Distribution` / `Bins` 输出 entropy、perplexity 和 active ratio，
  `Bins` 额外输出 occupied/dead count 与 ratio，`Profile` 输出各 coordinate bin 的 count/mean/std。

这里的提交窗口在 train 是一个 optimizer step，在 validation 是一个 epoch；不是 `every_n_steps` render
interval。`OutputObservation` / `ForwardEvent` 的 histogram 类型使用 `Reduction.Merge + EMAHistory`。
`Mean` 不推断 batch/count 权重；一个 mapping spec 的所有 value 必须兼容同一个 reduction。`StateObservation`
每个边界只 poll 一次，按值类型自然采用 snapshot 或 merge。

callback 默认只在 global rank 0 的 `on_fit_start` 扫描 `pl_module.named_modules()`；
`sync_distributed=True` 时才在所有 rank 扫描和注册。module class 在实现旁通过唯一
`observation.registry.register(ModuleType, specs)` 声明完整 observation contract；registry 按 exact
concrete type 查找，不保存实例或 tensor，deepcopy 直接复用 class declaration。当前 ADT、MoE 和六个
公开 quantizer 都已注册默认 observation。未挂 callback 时，构造器调用
`observation.registry.recommend(self)` 的推荐 module 首次 training + grad-enabled 普通调用会发出
一次 `RecommendationWarning`；父 module 声明 `covers_descendants=True` 时不会再提示或观察内部 stage。
`BackwardObservation` 当前是 TODO 占位；callback 发现后发出一次
`BackwardObservationWarning` 并跳过，不注册 backward hook 或修改 autograd graph。需要观察 loss 对
参数的梯度时先使用 `GradientProbeLoggerCallback`。

声明按来源分为 `StateObservation`、`OutputObservation` 和 `ForwardEvent`：State 在 optimizer boundary
读取 module 自然持有的状态；Output 从普通 `module(...)` 的 inputs/kwargs/output 提取；Event 在 forward
局部量产生处显式调用 `observation.emit(module, name, value)`。不得为了 logging 保存 `self.last_*`，也不
提供 observer-enabled 分支来改变 forward、autograd 或各 rank 的 attention kernel。

`every_n_steps` 控制 render 和无 history 的 State/Output/Event collector，默认每 100 个 optimizer step 输出一次；
State/Output collector 在 cadence 外不调用，ForwardEvent 的 `emit()` 仍由 module 执行，但 callback 直接丢弃 payload。
需要 EMA 的 Distribution/Profile/Bins 在被跳过的 step 仍更新状态。传 `None` 会关闭 render 和无 history
collector，但保留 Distribution/Profile/Bins 的 EMA 更新。每次 render 同时输出 `module/observation_time`：它是当前 interval
内全部 observation collector、状态更新、归约、校验、histogram writer 和 Heatmap image writer 的 host
wall time，按实际 optimizer step 数取平均。它不额外同步 CUDA，也不包含最终 scalar `log_dict()` submission。
callback 级 `half_life_steps` 会覆盖 spec 自带的 `EMAHistory`。TensorBoard 等 backend 支持
`add_histogram_raw` 时同时写 step/EMA histogram；不支持时仍保留 scalar summary。
`max_dense_bins` 在分配状态前限制 `[groups, max_bins]` 的 dense 大小，mixed-radix product codebook
超过限制会明确报错，调用方应改用 factorized observation 或显式提高上限。
Heatmap 只在 render step 由 global rank 0 生成独立 CPU image，并通过 `add_image(..., dataformats="HW")`
写出；backend 不支持 image 时明确报错。distributed 模式先完成 schema preflight，并同步 collector、image
和 histogram backend 错误。`max_heatmap_elements` 超限时要求 module 选择更小的二维语义 view，不自动
flatten、resize 或 sample。

默认 `sync_distributed=False`，只维护并输出 global-zero rank 的本地诊断。train 和 validation 分别维护
pending state 与归约时钟；开启 distributed 同步后 callback 会先同步
collector/record 错误，再确认所有 rank 拥有相同累计 observation key/schema/source identity，并在该
optimizer step 产生相同 emitted key 集合，之后才 sum sufficient stats；条件 module 调用或动态 mapping
必须在各 rank 保持一致，否则同步失败并显式报错。reduction 也是 distributed schema 的一部分；
`Last` Curve 先保留各 rank 的本地最后一次，再对 rank-local last 求均值。
内部 state key 与 Curve、step/EMA histogram scalar、raw histogram 的最终 logger tag 都要求全局唯一；
不同 module/spec 路径归一化到同一个实际 tag 时会立即报错，不会静默覆盖。

checkpoint 只保存已完成 optimizer step 的 Distribution/Profile/Bins EMA、稳定的 module/spec/item source
identity 和 step metadata，不保存 pending microbatch、Curve、Heatmap 或 raw forward output。恢复时 `tag` 作为
callback state identity 必须保持稳定，`half_life_steps` 与 `sync_distributed` 必须兼容；
`every_n_steps` 可以调整，`max_dense_bins` 可以调整到仍能容纳已保存 schema 的值。

直接调用 `.forward()` 会绕过 `OutputObservation` hook，但其中显式执行的 `emit()` 仍会发布
`ForwardEvent`，`StateObservation` 也仍会在 step boundary 读取；领域方法是否 emit 由 module 自己的
公开 contract 决定，不提供 module-specific callback 兜底。当前提交边界基于 train batch end 的
`global_step` 变化，因此支持常规自动优化和梯度
累积；同一个 train batch 内执行多个 optimizer step 的 manual optimization 会 fail-fast。

更完整的 registry、类型和 hook 边界见
[`anytrain.observation` 设计](modules/observation.md)。module 外计算的 aux loss、最终 evaluator、性能指标
和 loss-to-parameter gradient 继续使用各自生命周期入口，不放入 observation registry。

## Optimizer / Scheduler

如果 optimizer 作为 `pl_module` 依赖传入，推荐传入优化器工厂：

```python
class MyPLModule(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
```

也可以完全不配置 optimizer，直接在下游 `configure_optimizers()` 返回 Lightning 接受的原生结构：

```python
optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
return {
    "optimizer": optimizer,
    "lr_scheduler": {
        "scheduler": scheduler,
        "interval": "step",
    },
}
```

如果有多组 optimizer、多模块参数组、手动优化或更复杂的 scheduler 配置，下游 `configure_optimizers()` 直接返回 Lightning 原生结构即可。

## Logging Mixin

`LightningLogMixin` 提供几个适合在下游 `LightningModule` 里直接调用的小工具：

```python
from anytrain.lightning import LightningLogMixin


class MyPLModule(LightningLogMixin, pl.LightningModule):
    def training_step(self, batch, batch_idx):
        ...
        self.log_prefixed_dict("train/metric", metrics, on_step=True)
        return loss
```

当前能力：

- `log_prefixed_dict(prefix, values, **kwargs)`：给 dict key 统一加前缀后调用 Lightning 原生 `log_dict()`。
- `log_audio(tag, audio, sample_rate=..., step=None, rank_mode="zero")`：通过当前 `trainer.loggers` 写入 audio。
- `log_figure(tag, figure, step=None, rank_mode="zero")`：通过当前 `trainer.loggers` 写入 figure。

`audio` 和 `figure` 目前支持 `TensorBoardLogger`。如果当前 Trainer 没有 logger，或 logger backend 不支持对应能力，会直接抛出明确错误。

媒体日志默认只在 global rank 0 写入。调试分布式 rank 差异时，可以传 `rank_mode="all"`；此时所有 rank 都会写入，且分布式场景下 tag 会自动加上 `rank={global_rank}/` 前缀。

## Root

运行目录由下游创建 Trainer 时显式设置，例如：

```python
trainer = pl.Trainer(default_root_dir="outputs/my_project/debug")
```

logger backend 不再由 `anytrain.lightning` 自动创建；需要自定义 logger 时，直接通过 Lightning 原生 `trainer.logger` 配置或 Python 代码传入。

## 调试能力

`LossSummaryCallback` 从 mapping 形式的 training outputs 收集 `loss_key` 和 `loss_names` 显式选择的
0-D Tensor；任意选中值缺失、不是 Tensor 或不是 scalar 时，当前 batch 不写入任何窗口。
batch hook 只保留 detached device scalar，不在每个 microbatch 上转成 Python float；每个指标保存最早和
最近各 `window_capacity` 个值以及完整 observation count，因此内存不会随训练步数增长。默认
`window_capacity=20`；`report()` 默认使用配置的 capacity。已经超出保留范围的显式窗口会明确报错。

分布式训练在所有 rank 都执行的 `on_train_end()` 中先校验 metric 名称集合、window capacity 和
observation count 一致，再归约 first/last 窗口。单进程、外部启动的 DDP 和 Lightning subprocess DDP 可以在
`Trainer.fit()` 返回后调用 `report()`；global rank 0 得到的是跨 rank 均值，而不是 rank-local 结果。
如果 fit loop 完全跳过且没有 observation，`on_fit_end()` 只确认空报告，不会在 Lightning teardown 后
发起 distributed collective。
`ddp_spawn` / `ddp_fork` 的启动进程不会由 Lightning 恢复 callback 实例状态，使用这些 launcher 时应在
worker callback hook 调用父类同步后消费 `report()`；启动进程读取会明确报错。`report()` 在训练结束同步
前同样不可用。`values` 仅保留为有界、按需转到 host 的兼容只读视图，不应用作完整训练历史。

`GradientProbeLoggerCallback` 通过必填的 `loss_groups: GradientLossGroupsFn` 显式获取当前
loss graph，不会在 callback 内隐式探测 `pl_module` 方法。已有模块若实现了
`current_loss_outputs()` 或 `current_gradient_loss_groups()`，可以显式传入兼容 adapter：

```python
from anytrain.lightning import GradientProbeLoggerCallback, gradient_loss_groups

callback = GradientProbeLoggerCallback(
    comparisons=comparisons,
    probes=probes,
    loss_groups=gradient_loss_groups,
)
```

新代码也可以直接传入项目自己的 provider；它必须返回 `group -> scalar Tensor mapping`，其中默认
group 名通常是 `batch`，而且这些 Tensor 必须仍连接需要探测的 autograd graph，不能来自 observation
或 detached logging state。这样 loss graph 的所有权留在下游装配层，callback 只负责梯度观测。

`DebugCallback` 在调用方显式加入 callback 列表时启用。全部初始参数在 train start 检查一次；每次
backward 后只重复检查可训练参数与现有梯度，因此冻结参数不会反复扫描。正常路径按 device/dtype 用 foreach norm 聚合后再做一次
主机判定，只有发现异常时才逐 tensor 定位。

`OOMDiagnostics` 在调用方显式加入 callback 列表时启用。它记录 train/validation batch 当前的
Lightning phase，并在 Torch OOM 或兼容的 `RuntimeError("out of memory")` 上输出错误、rank、CUDA
memory 和调用上下文。下游通过 `input_reporter` 提供 JSON-compatible 的 batch/request 摘要；callback
不解释 batch schema，不重试或拆分 batch。生成或评估路径可以用 `annotate()` 附加调用点上下文，再用
`report_oom()` 把上下文交给已安装的 callback。

`PerformanceCallback` 在调用方显式加入 callback 列表时启用。它记录：

- `perf/model_params` 和 `perf/model_trainable_params`。
- `perf/model_flops_per_step` 和窗口值；固定计算量由下游直接传入，变长 batch 通过
  `FlopsProvider` 按 train batch 提供并在 optimizer step 边界聚合。
- `perf/hardware_peak_flops`，默认按设备型号和 compute dtype 查表，必要时由 job 覆盖。
- `perf/step_time`、`perf/step_time_window` 和 `perf/mfu`。

短性能探测可设置 `stop_after_measurement=True`：callback-relative warmup 后完整测量
`measure_window_steps` 个 optimizer update，记录最终窗口并请求 Trainer 停止。`report()` 返回不持有
输出路径的 `PerformanceReport`，其中包含窗口 optimizer step 数和同名 `perf/*` metrics。

代表性输入、训练 step FLOPs 口径属于下游任务语义，不由 callback 猜测。有效 audio 秒 / token
数等 unit 进度由 `anytrain.lightning.schedule` 的 `UnitClock` 显式提供，并由 unit runtime
记录 `progress/*`、padding 和 `{unit}_per_second`；DDP unit throughput 使用各 rank valid unit 总和
除以当前 optimizer update 的最慢 rank elapsed，窗口逐 update 累加最慢 rank elapsed。动态 FLOPs
窗口按 `sum(flops) / sum(time) / peak_flops` 计算；梯度累积按 optimizer `global_step` 形成
measurement。DDP provider 返回 local-rank FLOPs，
callback 统一聚合 rank 工作量、最慢 rank 时间和各 rank 峰值算力。`sync_distributed=True` 时，
每个 batch timer 启动前先 barrier，使上一 batch 的 rank-local callback 工作不会错开下一步计时
起点；barrier 自身不计入 step time。完整输入协议和边界见
[`docs/modules/perf.md`](modules/perf.md)。

如果参数或梯度中出现 NaN 或 Inf，它会打印第一个异常项的 name、index、value、shape、dtype、device 并直接抛错，避免继续写坏 checkpoint 或污染日志。

## Checkpoint 保存

`ModelCheckpoint` 继承 Lightning 原生 `ModelCheckpoint`，构造参数沿用原版接口，并在末尾增加
`async_save: bool = True` 和 `max_pending_async_saves: int = 1`。

默认 `async_save=True` 时，rank 0 会先把 checkpoint 写入本机临时目录，再用后台单线程队列复制到目标路径，适合目标目录挂在 NFS 等慢文件系统上的场景。持有完整本地 source 的复制任务达到
`max_pending_async_saves` 上限时，下一次保存会先等待最老复制任务；保存路径会在创建下一份完整
本地文件之前施加该背压。删除任务继续进入同一个单线程队列来保持 top-k 顺序，但不占复制 backlog
容量，也不会迫使刚提交的复制立刻转为同步等待。默认上限 1，因此慢复制不会让完整 checkpoint 在
本地无界积压；需要用有限本地空间换取更深 I/O overlap 时可以显式调高。传入 `async_save=False` 时，它的保存、top-k、
`save_last`、删除和 logger 通知行为保持原生同步逻辑。

异步队列串行处理目标路径复制和 top-k 删除，避免旧 checkpoint 的后台复制晚于删除完成而重新写回。后台复制会先写目标目录下的 `.part` 文件，再 `os.replace()` 到最终路径，减少半写入文件被观察到的风险。`save_last="link"` 会在持久目标 checkpoint 发布成功后原子替换 `last.ckpt`，而不是链接会被清理的本地临时 source；不能创建 symlink 的平台同样在主发布成功后执行 Lightning 的文件复制语义。复制期间已有 `last.ckpt` 保持可读；首次保存则直到主目标发布成功才创建。旧 top-k checkpoint 的删除还会等待本轮 last 更新成功，因此任一发布失败都保留上一份可读链接及其目标。`wait_async_saves()`、正常 fit end 和 exception drain 返回后保证新 last 已发布并可读。

当同一次回调既保存 top-k checkpoint 又设置 `save_last` 时，异步模式只调用一次 `trainer.save_checkpoint()`：先发布持久 top-k 文件，再从该文件原子更新或链接 `last.ckpt`。这同样适用于 `save_on_exception=True`；`save_top_k=0` 则只序列化 `last.ckpt` 一次。

使用 `async_save=True, save_last="link"` 从当前 `last.ckpt`（或它指向的同一文件）恢复时，如果本次
fit 没有进入任何 train batch，callback 会继续复用这个仍然存在的 `last.ckpt`，不在 train end 再
序列化一份完全相同的 checkpoint。跳过时 global rank 0 会输出包含 checkpoint 路径和零 batch 原因的
`Skipping last checkpoint save` 日志。只要进入过一个 train batch，就恢复原有的 train-end 保存语义；
从其它 checkpoint 恢复、恢复路径缺失、目录发生变化、`async_save=False` 或使用布尔 `save_last` 时
也保持 Lightning 原有行为。DDP 下是否复用由 global rank 0 检查并广播，所有 rank 进入相同分支。

第一版只支持本地文件系统路径，包括挂载到本机的 NFS；不支持 `s3://`、`gs://` 等远端 URI。后台任务错误会在下一次 checkpoint 操作、`on_fit_end()`、异常处理保存后或用户显式调用 `wait_async_saves()` 时抛出。DDP 下一次保存会先由 global rank 0 完成 backlog/error preflight，再把结果同步给所有 rank；发生 prior async error 时所有 rank 都在进入 `trainer.save_checkpoint()` collective 前失败，rank 0 保留原始异常 cause。`on_fit_end()` 和 `on_exception()` 的后台任务 drain 同样由 rank 0 等待后同步结果，避免训练收尾阶段各 rank 分叉。

Python 入口示例：

```python
from lightning import pytorch as pl
from anytrain.lightning import DebugCallback

trainer = pl.Trainer(callbacks=[DebugCallback()])
```

固定单个 batch 训练属于 dataloader / data module 的采样策略，不放在 `anytrain.lightning` 里。

## Batch 边界

`anytrain.lightning` 不解释 batch。

batch 可以来自：

- 下游 LightningDataModule。
- 普通 PyTorch DataLoader。
- `anydataset` 产生的 dataclass batch。
- 下游自定义对象。

`training_step` 中如何读取 batch 完全由下游 pl module 决定。

## 不做的事

`anytrain.lightning` 不做：

- 组件注入封装或组件加载协议。
- Trainer callback registry 或自动装配；module observation 的 type registry 是独立声明协议。
- 替用户写 task-level 训练 step。
- 替用户选择 optimizer / scheduler。
- model zoo。
- 自定义 `LightningModule` 魔法基类。
