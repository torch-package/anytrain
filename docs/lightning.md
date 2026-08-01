# Lightning

## 定位

`anytrain.lightning` 是面向普通 LightningModule 的核心运行时层。训练入口由下游项目自己维护；这个子包只提供 task-agnostic 的 runtime helper、logging mixin 和调试 callback。

这个包就是服务 Lightning 的，所以 `torch` 和 `lightning` 属于默认测试环境，不再把 Lightning 当成可跳过依赖。

它负责：

- LightningModule 侧的轻量 logging helper。
- Trainer 主 logger 的 text/scalar/audio/histogram experiment 能力访问。
- validation 指标按实际 observation count 做 epoch/DDP 加权聚合，并记录可恢复的指标历史。
- 训练 outputs 里的 total loss 和命名 loss item 窗口汇总 callback。
- 训练 outputs 里的命名 loss item 按下游 label 分组记录 callback。
- 基于命名 `LossItem` 的梯度关系、全局梯度范数和 time-bucket loss 观测 callback。
- 训练 outputs 中显式声明的 `QuantizeOutput` usage、perplexity、dead code 和 active codebook 观测 callback。
- non-finite loss 检查等训练调试 callback。
- NFS 等慢文件系统场景下可选的异步 checkpoint 落盘 callback。
- 训练效率观测 callback，记录参数量、step time、硬件峰值算力和 MFU。

它不负责：

- 替用户实现具体 `training_step`、`validation_step`、`test_step`。
- 替用户决定 batch schema。
- 替用户决定 optimizer / scheduler。
- 内置模型 zoo、组件加载协议或单模型快捷封装。

下游项目定义自己的 `pl_module`，直接继承 Lightning 原生 `LightningModule`，并按需组合 `anytrain` 提供的 loss、evaluator 或 framework 组件。

## Core 与 Optional

`anytrain` 可以提供训练组件，但按依赖层级拆分：

- core：`lightning`、`lightning.perf`、通用 loss/evaluator/optim/module 接口和 `module.idspace`。
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

配置形状和运行目录属于下游入口约定，不能藏进 `LightningModule` 继承链里。

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
        loss_result = self.loss_fn(y_hat, y)
        if isinstance(loss_result, tuple):
            loss, loss_details = loss_result
            self.log_dict({f"train/loss/{key}": value for key, value in loss_details.items()})
        else:
            loss = loss_result
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

`experiment.text()`、`scalar()`、`audio()` 和 `histogram()` 检查 Trainer 主 logger 的 experiment 是否
暴露对应 callable。没有 logger 或 backend 不支持时返回 `None`；helper 不决定 rank、记录 cadence、
tag 或 tensor 设备，下游 callback 继续拥有这些语义。

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
        self.log_prefixed_dict("train/loss", loss_details, on_step=True)
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

`LossSummaryCallback` 从 mapping 形式的 training outputs 收集 total loss 和命名 `LossItem`。
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

`DebugCallback` 在调用方显式加入 callback 列表时启用。全部初始参数在 train start 检查一次；每次
backward 后只重复检查可训练参数与现有梯度，因此冻结参数不会反复扫描。正常路径按 device/dtype 用 foreach norm 聚合后再做一次
主机判定，只有发现异常时才逐 tensor 定位。

`PerformanceCallback` 在调用方显式加入 callback 列表时启用。它记录：

- `perf/model_params` 和 `perf/model_trainable_params`。
- `perf/model_flops_per_step` 和窗口值；固定计算量由下游直接传入，变长 batch 通过
  `FlopsProvider` 按 train batch 提供并在 optimizer step 边界聚合。
- `perf/hardware_peak_flops`，默认按设备型号和 compute dtype 查表，必要时由 job 覆盖。
- `perf/step_time`、`perf/step_time_window` 和 `perf/mfu`。

代表性输入、训练 step FLOPs 口径属于下游任务语义，不由 callback 猜测。有效 audio 秒 / token
数等数据量通过 `DataThroughputCallback` + `DataUnitsProvider` 显式提供；padding 与
`{unit}_realtime` 由 valid/padded 同单位计算。动态窗口按 `sum(flops) / sum(time) / peak_flops`
计算；梯度累积按 optimizer `global_step` 形成 measurement。DDP provider 返回 local-rank FLOPs，
callback 统一聚合 rank 工作量、最慢 rank 时间和各 rank 峰值算力。`sync_distributed=True` 时，
每个 batch timer 启动前先 barrier，使上一 batch 的 rank-local callback 工作不会错开下一步计时
起点；barrier 自身不计入 step time。完整输入协议和边界见
[`docs/modules/perf.md`](modules/perf.md)。

如果参数或梯度中出现 NaN 或 Inf，它会打印第一个异常项的 name、index、value、shape、dtype、device 并直接抛错，避免继续写坏 checkpoint 或污染日志。

## Checkpoint 保存

`ModelCheckpoint` 继承 Lightning 原生 `ModelCheckpoint`，构造参数沿用原版接口，并在末尾增加 `async_save: bool = True`。

默认 `async_save=True` 时，rank 0 会先把 checkpoint 写入本机临时目录，再用后台单线程队列复制到目标路径，适合目标目录挂在 NFS 等慢文件系统上的场景。传入 `async_save=False` 时，它的保存、top-k、`save_last`、删除和 logger 通知行为保持原生同步逻辑。

异步队列串行处理目标路径复制和 top-k 删除，避免旧 checkpoint 的后台复制晚于删除完成而重新写回。后台复制会先写目标目录下的 `.part` 文件，再 `os.replace()` 到最终路径，减少半写入文件被观察到的风险。`save_last="link"` 保持 Lightning 原生逻辑；如果希望最后一份 checkpoint 也走异步文件复制，优先使用 `save_last=True`。

第一版只支持本地文件系统路径，包括挂载到本机的 NFS；不支持 `s3://`、`gs://` 等远端 URI。后台任务错误会在下一次 checkpoint 操作、`on_fit_end()`、异常处理保存后或用户显式调用 `wait_async_saves()` 时抛出。

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
- callback registry。
- 替用户写 task-level 训练 step。
- 替用户选择 optimizer / scheduler。
- model zoo。
- 自定义 `LightningModule` 魔法基类。
