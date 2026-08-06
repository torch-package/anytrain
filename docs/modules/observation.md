# `anytrain.observation` Design

## 定位

`anytrain.observation` 定义普通 `torch.nn.Module` 可声明的训练观测契约。它只依赖 PyTorch，
不持有 logger、Trainer 或 LightningModule；Lightning 生命周期由
`ObservationCallback` 适配。

唯一声明来源是进程级 `observation.registry`。module 实现文件在 class 定义完成后，用
`registry.register(ModuleType, specs)` 注册该 concrete type 的完整 immutable contract；构造器末尾调用
`registry.recommend(self)` 安装一次性的使用提示。contract 含 `recommended=True` spec 时必须调用；否则无需
调用。registry 只保存 module class 和无状态 collector，不保存
module 实例、tensor、forward output 或 autograd graph。deepcopy 直接复用 class declaration，spawn/DDP 进程
通过正常 import 重建同一 registry。

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
        ...


def _collect_router(module, inputs, kwargs, output):
    del module, inputs, kwargs
    return {"output_mean": observation.Curve(output.float().mean())}


observation.registry.register(
    Router,
    (
        observation.ForwardObservation(
            "routing",
            _collect_router,
            recommended=True,
        ),
    ),
)
```

## Module author 约定

设计或修改一个 `nn.Module` 的 forward contract 时，必须同时审查其输出和内部运行状态中是否存在值得训练期
观察的稳定中间量。有则在 module 实现旁注册；没有则可以不注册，但应是明确的设计结论。

registry + `ObservationCallback` 是 anytrain 内 module 中间变量 logging 的唯一入口。不得为具体 module 新增
logger callback、monitor adapter、training-output 自动发现或额外 forward。collector 必须只读，不修改
output/state；scalar 使用 `Curve`，明确的一维连续 view 使用 `Distribution`，离散 codes 使用 `Bins`。多维
连续 tensor 必须由 module 先选择有语义的 summary 或一维 view，不能自动 flatten。module 外计算的 aux loss、
最终 evaluator、性能指标和 loss-to-parameter gradient 不属于中间变量入口。

## 日志值

- `Curve`：0-D real Tensor。一个 optimizer step 内多次调用取 mean，跨 step 的历史由 logger
  曲线本身保存，不维护 EMA。
- `Distribution`：1-D floating Tensor，并显式提供固定 histogram edges。多维连续 tensor 不会
  自动 flatten；调用方必须先选择明确的 view。
- `Bins`：任意 rank 的离散 observation Tensor。单 bin space 时所有维度都是 observation axis；
  多 bin space 时通过 `group_dim` 指出 group axis，并为每组提供固定 `sizes`。可选
  `active_mask` 与 values 同 shape。没有 mask 时只把 `-1` 视为 inactive sentinel；其它负值或超出
  size 的 active code 都会报错。

`Distribution` 和 `Bins` 在 hook 内立即转换成固定大小的 histogram sufficient statistics，
不保留 forward output 或 autograd graph。历史使用 `EMAHistory(half_life_steps=...)`，日志明确区分
`step` 与 `ema`；entropy、perplexity 和 active ratio 都从 step/EMA counts 现算，不对派生指标做
EMA。`Bins` 还统一输出 occupied/dead count 与 ratio；EMA scope 的 occupied 表示 EMA mass 仍为正的 bin。

## Module spec

当前 spec 包括：

- `ForwardObservation`：从 module forward 的 inputs/kwargs/output 提取 observation。
- `BackwardObservation`：当前只保留为后续只读 backward observation 的占位声明。
  `ObservationCallback` 发现后发出一次 `BackwardObservationWarning` 并跳过，不注册 backward
  hook，也不修改 module output 或 autograd graph。当前需要查看 loss 对参数的梯度时，使用
  `anytrain.lightning.GradientProbeLoggerCallback`。
- `PollObservation`：在 optimizer step 完成时读取 module 已维护的状态。

`covers_descendants=True` 表示父 module 的观测已经覆盖该子树。Residual quantizer 使用这个规则，
避免同时记录外层组合结果和内部 stage quantizer，也会移除内部 stage 的推荐 hint。

注册的 `ForwardObservation` / `PollObservation` 设置 `recommended=True`，并在 module 构造器调用
`registry.recommend(self)` 时，会安装一次性
training forward hint；尚未实现的 `BackwardObservation` 不参与通用推荐或子树 hint 覆盖。没有
observer 时，首次 training + grad-enabled 调用会发出 `RecommendationWarning`，提示加入
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
module path 绑定 hook。默认只有
global rank 0 扫描并注册 forward hook；`sync_distributed=True` 时才在所有 rank 注册并同步状态。每个
需要 EMA history 的 Distribution/Bins 每个 train batch 都收集；无 history 的 Curve/collector 只在即将
render 的 optimizer step 执行，避免 cadence 之外计算昂贵派生指标。只有 `trainer.global_step` 前进时才提交
optimizer-step snapshot 和更新 EMA；`every_n_steps` 控制 render，默认每 100 个 optimizer step 输出一次，
传 `None` 完全关闭 render 和无 history collector。EMA state 随 callback checkpoint 保存，
pending microbatch state、Curve 和 hook 不保存。每个 histogram 同时保存稳定的
`module_path/spec_name/item` source identity，恢复时不靠 slash prefix 猜测 collector；`tag` 是 callback
identity。render cadence 可以改变，half-life 和 distributed 语义必须兼容。

每次 render 还会在同一次 `log_dict()` 中输出 callback 自身的 `module/observation_time` 曲线（`module`
由 `tag` 参数决定）。它累计该 render interval 内所有 module collector、state/EMA 更新、distributed
归约、校验和 histogram render 准备的 host wall time，再按实际 optimizer step 数取平均，单位为秒。
最终 scalar logger submission 不计入该值，也不会为了计时额外执行 CUDA synchronize；因此它表达训练在
observation callback 入口实际承担的 host stall，不是隔离后的 GPU kernel 时间。该 interval 计时不进入
checkpoint；`sync_distributed=True` 时各 rank 各自计时，默认可见的是 logger owner rank 的值。

`sync_distributed=False` 默认只维护并输出 global-zero rank 的本地诊断。显式开启时，每个已发现 observer 必须
在所有 rank 以一致结构调用；callback 先 all-gather 累计 key/schema、稳定 source identity 和当前
optimizer step 实际 emitted key 集合的 signature，确认一致后再 sum counts/sums、更新相同 EMA 和计算
派生指标，避免历史上曾出现过的 key 在当前 step 只由部分 rank 再次产生，或不同 source 恰好拼成同一
state key 时静默通过。

不同 collector/value 组合不得归一化到同一个内部 state key；Curve、histogram step/EMA scalar 以及 raw
histogram 实际会写出的完整 logger tag 也必须全局唯一。冲突在首次 observation 或 checkpoint 绑定时
明确报错，不会由 `dict.update()` 静默覆盖。

`max_dense_bins` 默认限制 dense histogram 大小，避免 mixed-radix product codebook 意外申请巨型
`[groups, bins]` 状态。超过上限会明确报错；这类 quantizer 应提供 factorized observation，或由调用方
确认成本后显式提高上限。

PyTorch forward hook 只覆盖普通 `module(...)` 调用；直接调用 `.forward()` 或领域方法（例如
`quantizer.quantize()`）不会触发自动观测。

当前 optimizer-step 提交使用 train batch end 的 `global_step` delta，支持自动优化与梯度累积；一个
train batch 内多个 manual optimizer step 无法无歧义拆分 pending state，会明确报错。非法
Distribution/Bins 数值在完成同一步的 distributed reduce 后立即报错，不受 render cadence 影响。
