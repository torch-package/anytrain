# `anytrain.observation` Design

## 定位

`anytrain.observation` 定义普通 `torch.nn.Module` 可声明的训练观测契约。它只依赖 PyTorch，
不持有 logger、Trainer 或 LightningModule；Lightning 生命周期由
`ObservationCallback` 适配。

源码 ownership：

```text
src/anytrain/observation/
  __init__.py   # 稳定公开入口与 compatibility-only attributes
  types.py      # public observation value/spec/reduction 类型
  _binding.py   # process registry、dynamic context 与 callback transport runtime
```

`types.py` 中的类型从 package root 重导出；普通调用方使用 `anytrain.observation` 即可。
`_binding.claim/release/subscribe_event` 是 Lightning callback 的私有 transport SPI，不是 module
author API，也不从 package root 导出。

唯一声明来源是进程级 `observation.registry`。module 实现文件在 class 定义完成后，用
`registry.register(ModuleType, specs)` 注册该 concrete type 的完整 immutable contract；构造器末尾调用
`registry.recommend(self)` 安装一次性的使用提示。contract 含 `recommended=True` spec 时必须调用；否则无需
调用。registry 只保存 module class、无状态 collector 和 event declaration，不保存 module 实例、tensor、
forward output 或 autograd graph。deepcopy 直接复用 class declaration，spawn/DDP 进程通过正常 import
重建同一 registry。

registry 按 exact concrete type 查找，不隐式继承父类声明。新增 subclass 时必须重新审查 forward contract
并显式注册；这能避免 subclass 改变 output/schema 后仍静默沿用父类 collector。同一个 class 只能注册一次，
spec name 归一化后必须唯一，collector 不能绑定到某个 `nn.Module` 实例。

```python
import torch

from anytrain import observation


class Router(torch.nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...
        observation.registry.recommend(self)

    def forward(self, logits):
        weights = logits.softmax(dim=-1)
        observation.emit(
            self,
            "attention",
            observation.Heatmap(weights[0], value_range=(0.0, 1.0)),
        )
        return weights


def _collect_router(module, inputs, kwargs, output):
    del module, inputs, kwargs
    return {"output_mean": observation.Curve(output.float().mean())}


observation.registry.register(
    Router,
    (
        observation.OutputObservation(
            "routing",
            _collect_router,
            reduction=observation.Reduction.Mean,
            recommended=True,
        ),
        observation.ForwardEvent(
            "attention",
            reduction=observation.Reduction.Last,
        ),
    ),
)
```

## Module author 约定

设计或修改一个 `nn.Module` 的 forward contract 时，必须同时审查其输出和内部运行状态中是否存在值得训练期
观察的稳定中间量。有则在 module 实现旁注册；没有则可以不注册，但应是明确的设计结论。

registry + `ObservationCallback` 是 anytrain 内 module 中间变量 logging 的唯一入口。不得为具体 module 新增
logger callback、monitor adapter、training-output 自动发现或额外 forward。module author 按值的来源选择：

- 自然存在于 module 上、会跨 forward 保留的运行状态用 `StateObservation`。不得为了 logging 新增
  `self.last_attention` 一类缓存。
- 能从普通 `module(...)` 的 inputs/kwargs/output 得到的值用 `OutputObservation`。
- 只存在于 forward 局部作用域、没有出现在调用边界的值用 `ForwardEvent`，并在产生处调用
  `observation.emit(self, name, value)`。

`OutputObservation` 和 `ForwardEvent` 注册时必须显式选择 `reduction`，不能由 callback 根据 tensor shape
猜测。`StateObservation` 每个提交边界只 poll 一次，因此不增加 reduction 字段。

调用方可以用动态 context 给一次完整 module 调用树增加公共前缀：

```python
from anytrain import observation

with observation.context("reconstruction"):
    loss = loss_fn(prediction, target)
```

context 使用 `ContextVar` 传播并支持嵌套；父 module、child module 的 `OutputObservation` 和 forward
内部发布的 `ForwardEvent` 都读取同一个调用上下文。Lightning transport 再添加 stage 前缀，因此上例在
train/validation 分别记录到 `train/reconstruction/...` 和 `val/reconstruction/...`。context 在退出
`with` 后立即恢复，不写入 module 状态，也不保存所谓“上一次 forward”值。validation 没有显式 context
时默认使用 `val/...`。

collector 和 event 都必须只读，不修改 output/state，也不能参与 forward 返回值。scalar 使用 `Curve`，明确的
一维连续 view 使用 `Distribution`，一维数值按另一条连续坐标分段统计时使用 `Profile`，离散 codes 使用
`Bins`。多维连续 tensor 不能自动 flatten；真正需要看图时，module 必须先选定 sample/head/crop，得到语义明确的
二维 `Heatmap`。module 外计算的 aux loss、最终 evaluator、性能指标和 loss-to-parameter gradient 不属于
中间变量入口。

## Reduction

reduction 折叠的是同一 source 在一个提交窗口内的重复 observation：train 窗口是一个 optimizer step，包含
gradient accumulation 的多次 forward；validation 窗口是一个 validation epoch。它不是
`every_n_steps` 控制的 render interval，也不是 DDP rank 间的 reduction。

- `Reduction.Last`：只支持 `Curve` / `Heatmap`，保留窗口内最后一次 emission。
- `Reduction.Mean`：只支持 `Curve`，按 emission 等权求算术平均；不根据 batch size、mask count 或 tensor
  shape 推断权重。
- `Reduction.Merge`：只支持 `Distribution` / `Profile` / `Bins`，合并 fixed-size sufficient statistics。

`Merge` 必须同时声明 `EMAHistory`；`Last` / `Mean` 不允许 history。一个 spec 返回 mapping 时，reduction
分别作用于每个 item，但整个 spec 只有一个 reduction，因此所有 value 都必须与它兼容；同一窗口内 mapping
keys 应保持稳定。需要同时记录 scalar 和 histogram 时拆成两个 spec。

`StateObservation` 在边界只读取一次：`Curve` / `Heatmap` 自然按 snapshot 处理，
`Distribution` / `Profile` / `Bins` 自然 merge；history 约束仍在首次读到值时校验。

## 日志值

- `Curve`：0-D real Tensor。重复值由 spec 显式选择 `Last` 或 `Mean`；跨提交窗口的历史由 logger 曲线本身
  保存，不维护 EMA。
- `Distribution`：1-D floating Tensor，并显式提供固定 histogram edges。多维连续 tensor 不会
  自动 flatten；调用方必须先选择明确的 view。
- `Profile`：等长、非空、同 device 的 1-D floating `values` / `coordinates`，并显式提供固定 coordinate
  edges。每个 coordinate interval 记录 value 的 count、mean 和 std，同时保留 step 与 EMA sufficient
  statistics。典型用途是 loss 随 timestep、长度或噪声水平的分段变化；它取代为某个具体字段定制的
  time-bucket callback。
- `Bins`：任意 rank 的离散 observation Tensor。单 bin space 时所有维度都是 observation axis；
  多 bin space 时通过 `group_dim` 指出 group axis，并为每组提供固定 `sizes`。可选
  `active_mask` 与 values 同 shape。没有 mask 时只把 `-1` 视为 inactive sentinel；其它负值或超出
  size 的 active code 都会报错。
- `Heatmap`：非空 2-D floating Tensor。module 必须先选择 sample、head 和空间范围，不能把任意高维
  tensor 直接交给 logger。`value_range` 固定时不同 step 的颜色可比较；传 `None` 时每张图按自身有限
  min/max 缩放。同一 key 的 shape、dtype 和 `value_range` 在一次运行内必须稳定。Heatmap 不维护 EMA，
  所属 spec 使用 `Reduction.Last` 且 `history` 必须为 `None`，需要同时记录 histogram 时拆成另一个 spec。
  同一提交窗口同一 key 多次产生时保留最后一次。

`Distribution`、`Profile` 和 `Bins` 在 callback 消费时立即转换成固定大小的 sufficient statistics，
不保留 forward output 或 autograd graph。历史使用 `EMAHistory(half_life_steps=...)`，日志明确区分
`step` 与 `ema`；派生指标都从 step/EMA state 现算，不单独做 EMA。`Distribution` / `Bins` 输出
entropy、perplexity 和 active ratio；`Bins` 还输出 occupied/dead count 与 ratio，EMA scope 的 occupied
表示 EMA mass 仍为正的 bin。`Profile` 输出各 coordinate bin 的 count/mean/std 和整体 active ratio。

## Module spec

当前 spec 包括：

- `StateObservation`：在 optimizer-step hook 边界读取 module 已自然维护的状态。
- `OutputObservation`：从普通 module forward 的 inputs/kwargs/output 提取 observation。
- `ForwardEvent`：声明 forward 内显式发布的局部值。event name 必须匹配 exact concrete type 的 registry
  declaration；所有 rank 都应无条件执行同一 `emit()`，不能根据 observer 是否存在切换 forward 分支。
- `BackwardObservation`：当前只保留为后续只读 backward observation 的占位声明。
  `ObservationCallback` 发现后发出一次 `BackwardObservationWarning` 并跳过，不注册 backward
  hook，也不修改 module output 或 autograd graph。当前需要查看 loss 对参数的梯度时，使用
  `anytrain.lightning.GradientProbeLoggerCallback`。

`ForwardObservation` 和 `PollObservation` 只保留为 `OutputObservation` / `StateObservation` 的显式 import
兼容 alias；新代码和文档使用 canonical 名称。

`covers_descendants=True` 表示父 module 的观测已经覆盖该子树。Residual quantizer 使用这个规则，
避免同时记录外层组合结果和内部 stage quantizer，也会移除内部 stage 的推荐 hint。

注册的 `StateObservation` / `OutputObservation` / `ForwardEvent` 设置 `recommended=True`，并在 module 构造器调用
`registry.recommend(self)` 时，会安装一次性
training forward hint；尚未实现的 `BackwardObservation` 不参与通用推荐或子树 hint 覆盖。没有
observer 时，首次 training + grad-enabled 普通 `module(...)` 调用会发出 `RecommendationWarning`，提示加入
`ObservationCallback`；eval、no-grad、非零 DDP rank 和 `torch.compile` 图内不提示。callback
claim module 后会消费该 hint。hint 的 rank 探测是 best-effort，不能让 module forward 因未完整建立
process group 而失败；deepcopy 后复制的 hint 也会在首次有效调用或 claim 时自行移除。

## Lightning 入口

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

callback 在 `on_fit_start` 遍历 `pl_module.named_modules()`，按 concrete type 查询 registry，并用完整
module path 绑定 state/output/event collector。默认只有 global rank 0 绑定；`sync_distributed=True` 时才在
所有 rank 绑定并同步状态。train observation 按 optimizer step 归约，`every_n_steps` 控制 render；validation
observation 在 validation epoch 结束时独立归约并使用独立 EMA clock，不会和 train pending state 混合；
`every_n_steps` 只控制 train render，validation 始终按 epoch 提交。
每个需要 EMA history 的 Distribution/Profile/Bins 在对应 stage 的每次调用中收集；train 下无 history 的
State/Output collector 只在即将 render 的 optimizer step 调用，validation 则在当前 validation epoch 收集。
`ForwardEvent` 的 `emit()` 仍由 module 在所有 rank 无条件执行，callback 不采集时直接丢弃 payload。
EMA state 随 callback checkpoint 保存，pending microbatch/validation state、Curve、Heatmap 和 runtime binding
不保存。每个 histogram 同时保存稳定的 `stage/context/module_path/spec_name/item` source identity，恢复时不靠
slash prefix 猜测 collector；`tag` 是 callback identity。render cadence 可以改变，half-life 和 distributed
语义必须兼容。

每次 render 还会在同一次 `log_dict()` 中输出 callback 自身的 `observation/time` 曲线（固定为 `observation/time`，
不随 `tag` 参数变化）。它累计该 render interval 内所有 module collector、state/EMA 更新、distributed
归约、校验、histogram writer 和 Heatmap image writer 的 host wall time，再按实际 optimizer step 数取平均，
单位为秒。最终 scalar `log_dict()` submission 不计入该值，也不会为了计时额外执行 CUDA synchronize；因此它表达训练在
observation callback 入口实际承担的 host stall，不是隔离后的 GPU kernel 时间。该 interval 计时不进入
checkpoint；`sync_distributed=True` 时各 rank 各自计时，默认可见的是 logger owner rank 的值。

`sync_distributed=False` 默认只维护并输出 global-zero rank 的本地诊断。显式开启时，每个已发现 observer 必须
在所有 rank 以一致结构调用；callback 先同步 collector/record 错误，再 all-gather 累计 key/schema、稳定 source
identity、reduction 和当前 optimizer step 实际 emitted key 集合的 signature，确认一致后再 sum sufficient
stats、更新相同 EMA 和计算派生指标。`Last` Curve 先在每个 rank 保留本地最后一次，再对各 rank 的 last
求均值。这样单 rank collector 失败不会让其它 rank 继续进入不同的 collective；也避免历史上曾
出现过的 key 在当前 step 只由部分 rank 再次产生，或不同 source 恰好拼成同一 state key 时静默通过。

Heatmap 的 key、shape、dtype 和 `value_range` 也进入 distributed schema；像素不跨 rank 归约，只由
global rank 0 将提交窗口内最后一次有效 observation 转换为 detached、独立的 CPU float image，并通过 logger experiment 的
`add_image(..., dataformats="HW")` 写出。实际有 Heatmap 需要 render 但 backend 不提供 `add_image()` 时明确
报错；distributed 模式会先完成 schema preflight，再把 image/histogram writer 的 discovery、materialization
和 write 错误同步给所有 rank，避免单 rank 提前退出 collective。`max_heatmap_elements` 限制二维 view 的总元素数，超限时要求 module 显式选择
更小的语义视图，不自动 flatten、resize、crop 或 sample。

不同 collector/value 组合不得归一化到同一个内部 state key；Curve、histogram step/EMA scalar 以及 raw
histogram 实际会写出的完整 logger tag 也必须全局唯一。冲突在首次 observation 或 checkpoint 绑定时
明确报错，不会由 `dict.update()` 静默覆盖。

`max_dense_bins` 默认限制 dense histogram 大小，避免 mixed-radix product codebook 意外申请巨型
`[groups, bins]` 状态。超过上限会明确报错；这类 quantizer 应提供 factorized observation，或由调用方
确认成本后显式提高上限。

PyTorch forward hook 只覆盖普通 `module(...)` 调用，因此直接调用 `.forward()` 会绕过
`OutputObservation`；forward body 内显式执行的 `emit()` 仍可被 `ForwardEvent` 消费，`StateObservation`
也仍会在 step boundary 读取。`torch.compile` 捕获期间 `emit()` 直接 no-op，避免 logging side effect 改变
graph。observation 不提供 `enabled()` 分支；如果 flash/SDPA 路径不产生 attention map，必须由所有 rank
一致的显式模型配置决定是否改用会物化权重的实现，callback 不能隐式切换 attention kernel。

正常 Lightning 训练通过 `on_before_optimizer_step()` 在每个 optimizer step 边界提交 observation；
gradient accumulation 的多个 microbatch 仍属于同一个 optimizer step。`automatic_optimization=False` 的
manual optimization 也按每次 optimizer hook 独立提交，因此 discriminator/generator 等多个 step 不会合并。
如果运行环境没有触发该 hook，callback 会退回到 train batch end 的 `global_step` delta 兼容路径；该路径只能
精确处理一个 step，不能为同一 batch 中的多个 step 做归因。非法
Distribution/Profile/Bins 数值在完成同一步的 distributed reduce 后立即报错，不受 render cadence 影响。

## 验收

```bash
python -m ruff check src/anytrain/observation tests/test_observation.py \
  tests/test_observation_event.py tests/test_lightning_observation.py
basedpyright --pythonpath /path/to/python src/anytrain/observation \
  tests/test_observation.py tests/test_observation_event.py \
  tests/test_lightning_observation.py
python -m pytest -q tests/test_observation.py tests/test_observation_event.py \
  tests/test_lightning_observation.py
```

## 内部实现边界（2026-09）

Lightning transport 的状态、delta、schema、快照与历史编解码由
`lightning/callback/_observation_state.py` 集中拥有；writer 只消费快照。
`ObservationCallback` 保留挂接、阶段时钟、rank 预检/归约和 writer 错误同步。
历史 checkpoint v2/v3/v4 恢复规则及公开 observation 类型不变。
