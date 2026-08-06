# `anytrain.observation` Design

## 定位

`anytrain.observation` 定义普通 `torch.nn.Module` 可声明的训练观测契约。它只依赖 PyTorch，
不持有 logger、Trainer 或 LightningModule；Lightning 生命周期由
`ObservationCallback` 适配。

模块可以在构造时调用 `observation.register()` 绑定实例级 spec，也可以实现
`ObservationProvider.observation_specs()`，让声明在 deepcopy、spawn 和重新装配后仍可发现。
内置 quantizer 使用 provider 形式。实例级 spec 保存在 module 自身的普通 Python metadata 中，runtime
claim/hook 才使用弱引用状态，因此 bound collector 不会被全局 registry 反向保活。

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
EMA。

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

当前可执行的 `ForwardObservation` / `PollObservation` 设置 `recommended=True` 时会安装一次性
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

callback 在 `on_fit_start` 遍历 `pl_module.named_modules()`，用完整 module path 绑定 hook。默认只有
global rank 0 扫描并注册 forward hook；`sync_distributed=True` 时才在所有 rank 注册并同步状态。每个
train batch 都收集，但只有 `trainer.global_step` 前进时才提交一个 optimizer-step snapshot 和更新 EMA，
`every_n_steps` 只控制 render，传 `None` 完全关闭 render。EMA state 随 callback checkpoint 保存，
pending microbatch state、Curve 和 hook 不保存。每个 histogram 同时保存稳定的
`module_path/spec_name/item` source identity，恢复时不靠 slash prefix 猜测 collector；`tag` 是 callback
identity。render cadence 可以改变，half-life 和 distributed 语义必须兼容。

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
