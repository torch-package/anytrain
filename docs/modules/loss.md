# `anytrain.loss` Design

## 定位

`anytrain.loss` 提供下游 `LightningModule` 可直接组合的 task-agnostic loss。core loss
不解释 batch，也不绑定 audio、text 或 speech 领域；领域实现放在 optional 子模块。

优化目标和训练期诊断是两个接口：loss 只返回最终的 0-D `torch.Tensor`，诊断值通过
`anytrain.observation` registry 声明和记录。loss 不再返回 `details`，也不把日志状态塞进
训练输出。

## 公开结构

源码结构：

```text
src/anytrain/loss/
  __init__.py
  _group.py
  _observation.py
  _validation.py
  abc.py
  balancer.py
  codebook.py
  codec.py
  frame.py
  group.py
  temporal/
  spectral/
```

根包公开：

- `LossABC`
- `LossBalancerABC`
- `WeightedLossBalancerABC`
- `LossMapping`
- `MeanLossBalancer`
- `FixedWeightLossBalancer`
- `InterpolatedLossBalancer`
- `UncertaintyNormalization`
- `UncertaintyLossBalancer`
- `LossGroup`
- `MaskedFrameMSELoss` / `masked_frame_mse_loss`
- `MaskedCosineAlignmentLoss` / `masked_cosine_alignment_loss`
- `TeacherLoss` / `teacher_loss`
- `MaskedCodebookCrossEntropyLoss`
- `PackedCodebookLogits`

`LossItem`、`LossOutputs`、`LossDetails`、`LossResult` 以及
`combine_loss_outputs()`、`iter_loss_items()`、`loss_item_mean()` 等旧的行级组合 API
已经删除。训练 step 应该组合已经归约好的 scalar；需要跨任务组合时使用
`LossGroup` 和 balancer。

## Scalar loss protocol

`LossABC` 继承 `torch.nn.Module`。子类实现：

```python
def compute_loss(self, *args, **kwargs) -> torch.Tensor:
    ...
```

`forward()` 会校验并固定返回 0-D Tensor。向量 Tensor、tuple、mapping 或其它带诊断的
返回值都会立即抛错。`LossBalancerABC` 使用同样的 scalar 契约：它接受名称到 0-D
Tensor 的 mapping，并返回一个 0-D Tensor。

`LossGroup` 对每个 child 调用同一组输入，再把 child scalar 交给 balancer。默认的
`MeanLossBalancer` 求算术平均；`FixedWeightLossBalancer` 使用显式权重；
`UncertaintyLossBalancer` 维护可训练的 `log_var` 并按 uncertainty 权重组合。所有
balancer 的权重和参数都属于 module state，不会通过 loss 返回值暴露。

`WeightedLossBalancerABC` 是带稳定名称顺序和公开 `weights` 的 balancer 契约。
`weights` 始终表示当前实际施加给各 child loss、从而传给模型参数梯度的系数；
balancer objective 仍可额外包含只作用于自身参数的正则项。统一生效值入口只有
`balancer.weights`；interpolated balancer 的两个端点分别是
`balancer.fixed.weights` 和 `balancer.dynamic.weights`，不再提供平行的
`fixed_weights` / `dynamic_weights` 别名。

自定义 weighted balancer 调用 `super().__init__(loss_names)`，实现 `weights` 和
`compute_loss()`；如果静态配置会改变 checkpoint 的解释，覆写 `_checkpoint_config()`。

`FixedWeightLossBalancer(normalize_weights=False)` 默认执行显式 weighted sum；传入
`normalize_weights=True` 时只在构造阶段把固定权重除以其正的有限总和。固定权重不会在
forward 中隐式复制到 loss device 或降到 loss dtype；包含 balancer 的 module 必须按
PyTorch 常规方式整体移动或转换。

`UncertaintyLossBalancer` 令 `s_i = log_var_i`、`w_i = exp(-s_i)`。当
`normalization=UncertaintyNormalization.NONE`（配置字符串为 `"none"`）时返回
canonical uncertainty-style objective：

```text
sum_i(w_i * loss_i) + sum_i(s_i)
```

默认 `normalization=UncertaintyNormalization.DETACHED_SUM`（配置字符串为
`"detached_sum"`）时使用 detached-sum gradient normalization：

```text
Z = stop_gradient(sum_i(w_i))
total = (sum_i(w_i * loss_i) + sum_i(s_i)) / Z
```

这样 `log_var=0` 时 total 与算术平均同尺度，模型参数收到的 loss 权重和为 1，
而 `log_var` 梯度方向与未归一化 uncertainty objective 相同、仅共同缩放 `1 / Z`。
由于 `Z` 显式 detach，这是一种 surrogate gradient；其 autograd 梯度不等于把 forward
数值函数视为普通可微函数时的有限差分。一次 forward 只计算一次 `exp(-log_var)`。

### Fixed-to-dynamic transition

`InterpolatedLossBalancer` 从构造开始同时持有 fixed 和 dynamic 两个
`WeightedLossBalancerABC`，两侧 `loss_names` 及顺序必须完全一致。`blend=0` 使用 fixed
objective，`blend=1` 使用 dynamic objective，中间值执行线性插值：

```text
total = (1 - blend) * fixed(losses) + blend * dynamic(losses)
weights = (1 - blend) * fixed.weights + blend * dynamic.weights
```

这里必须插值完整 objective，而不是只插值权重，因为 uncertainty 等 dynamic balancer
还包含 `log_var` 正则项。dynamic 参数从构造时就进入 module tree、optimizer、DDP/FSDP 和
checkpoint；训练中不替换 balancer，也不临时创建参数。即使 `blend=0`，两侧仍执行 forward，
保持所有 rank 的计算图和参数使用方式稳定。

```python
from anytrain.loss import (
    FixedWeightLossBalancer,
    InterpolatedLossBalancer,
    UncertaintyLossBalancer,
)

balancer = InterpolatedLossBalancer(
    fixed=FixedWeightLossBalancer(
        {"reconstruction": 1.0, "perceptual": 3.0},
        normalize_weights=True,
    ),
    dynamic=UncertaintyLossBalancer(["reconstruction", "perceptual"]),
)

# 在 on_train_batch_start 等显式边界按 global_step 或 unit progress 更新。
balancer.set_blend(min(global_step / transition_steps, 1.0))
total = balancer(losses)
```

`set_blend()` 只接受 `[0, 1]` 内有限 host real。balancer 不在 forward 内统计调用次数，
因为 validation、梯度累积、closure、重计算和不同 rank 的条件调用都不能作为训练进度。
若 fixed 使用 `normalize_weights=True`、dynamic 使用默认 `DETACHED_SUM`，两端权重和都为 1，
过渡时不会额外改变模型梯度的整体尺度。分布式训练时每个 rank 必须在同一训练边界设置
相同的 `blend`。

`blend=0` 时 dynamic branch 仍在 autograd graph 中，因此 DDP 不会把其参数视为 unused；
但这只会产生零梯度，不等于 optimizer 意义上的冻结。Adam/AdamW 仍可能创建 state 并推进
parameter step，带 weight decay 或已有 momentum 时还可能改变参数。内置 uncertainty 的
`log_var` 从零开始，anytrain 默认参数分组会把它放入 no-decay，因而不会在初始零梯度阶段漂移；
若下游需要很长的纯 fixed 阶段，并要求 dynamic optimizer state 到启用时仍是 fresh，必须由
下游显式选择 optimizer-state reset 或其它 calibration 策略，loss 层不会隐式操作 optimizer。

带名称状态的 balancer 在 checkpoint extra state 中保存 loss 名称顺序；uncertainty
balancer 还保存 `normalization` 模式；interpolated balancer 保存 `blend`、两侧状态和静态
类型/config identity。加载时名称顺序或模式不一致会明确失败，避免位置向量
被静默解释成另一组 loss。interpolated checkpoint 会在复制任何 composite state 前递归预检
父级和两侧 metadata、state key 及 Tensor shape，配置或 schema 拒绝不会先局部覆盖
`blend` 或 child state。即使外层调用使用 `strict=False`，balancer 自身仍要求完整且
形状匹配的状态；`strict=False` 只放宽 balancer 之外的 module 字段。实例内部的
`loss_names` 是不可变 tuple。旧格式 checkpoint
若不含名称元数据，即使 `strict=False` 也不会按当前位置静默恢复；调用方必须先根据创建时
配置确认名称顺序并显式迁移。这是一次 checkpoint format breaking change。名称元数据使用
标准 PyTorch module extra state，适用于 `torch.save` 和 Lightning checkpoint；只接受纯
Tensor mapping 的 safetensors-only 保存链需要由调用方单独保存并恢复这部分元数据。

## Reduction semantics

每个 loss 在一次 `forward` 内完成自己的 logical reduction。下游必须把一次调用的输入
定义成完整的 reduction domain；scalar 返回后无法恢复原始 count。

- `masked_frame_mse_loss`：先在最后一个 feature 轴求每帧 MSE，再对所有 `mask=True`
  的 frame 求均值。每个有效 frame 权重相同。
- `masked_cosine_alignment_loss`：对每个有效 frame 求 cosine distance，再对有效
  frame 求均值。`detach_target=True` 时 target 不参与反向图。
- `teacher_loss`：cosine distance 和 Smooth L1 各自按有效 frame 求均值，再计算
  `cosine + smooth_l1_weight * smooth_l1`。
- `MaskedCodebookCrossEntropyLoss`：只对有效 frame 执行 CE；每个 codebook 先对
  有效 frame 求均值，最后对 codebook 求均值。已经压紧有效帧时，直接调用 module 并传入
  `PackedCodebookLogits(logits, labels)`，其中 logits 是每个 codebook 的 `[N, vocab]`
  Tensor，labels 是 `[N, Q]`。packed 输入不携带 batch row metadata，因为 scalar reduction
  只依赖完整有效 frame domain；普通输入和 packed 输入都使用标准 `module(...)` 调用。
- `SDRLoss`：先按样本计算 SDR/SI-SNR，再对 batch 求均值。

frame/codebook 的有效 reduction domain 为空时，loss 会抛出 `ValueError`，不会返回
`nan` 或静默的零值。

### 分片边界

如果不同 rank、数据 shard 或 gradient-accumulation microbatch 覆盖的有效 count 不同，
对各自返回的 scalar 直接求平均并不等价于全局 count-weighted mean。调用方必须满足
以下之一：让一次 loss 调用覆盖完整 logical domain；保证各 shard 的 count 相等；或在
训练编排层用调用方已知的 count 对目标显式缩放。loss API 不返回公开 count，也不尝试
在 callback 中猜测或补做跨 rank reduction。每个 child 负责自己的局部加权和 reduction；
`LossGroup` 与 `CodecLoss` 只组合 child 已经返回的 scalar。

## Observation registry

诊断的唯一入口是 `anytrain.observation` registry；不再有 loss-specific logger callback
或 `details` mapping。

已注册 loss module 的最终输出通过
`OutputObservation("loss", ..., reduction=observation.Reduction.Mean)` 记录为 `Curve`；logging-only scalar
diagnostics 的 `ForwardEvent` 同样显式使用 `Mean`。
`LossGroup` 只声明自己的 total；嵌套 child 若有稳定诊断，则由 child 自己声明，避免
group 复制或缓存 child 的中间状态。

推荐 observation 保持最小集合：

- `LossABC` 的具体原子 loss 默认只记录最终 `loss`；`LogMagnitudeLoss`、
  `SpectralRMSELoss` 和 `SDRLoss` 不重复记录可由最终 loss 直接推回的值。
- `LossGroup`、`STFTLoss`、`MelLoss`、multi-scale loss 和 `CodecLoss` 只记录自己的 total；
  term、scale 和 family 的 scalar 由正常调用的 child module 各自记录。
- `MaskedFrameMSELoss` 额外记录 `valid_frames`；`MaskedCosineAlignmentLoss` 记录
  `cosine` 和 `valid_frames`；`TeacherLoss` 记录 `cosine`、`smooth_l1` 和 `valid_frames`。
- `MaskedCodebookCrossEntropyLoss` 记录每个 codebook 的 CE 与 `valid_frames`；显式传入
  `include_top1=True` 时才额外记录逐 codebook top-1。
- `CompressedSpectrogramLoss` 记录内部相加的 `magnitude`、`real` 和 `imag` 三项；不记录
  spectrum、phase 或其它高维变换结果。

balancer 的持久属性通过 sibling `StateObservation` 读取：`weights/<loss>` 始终表示实际
生效权重，fixed balancer 额外提供 `fixed/<loss>`，dynamic balancer 额外提供
`dynamic/<loss>`。interpolated balancer 同时提供 `weights`、`fixed`、`dynamic` 和 scalar
`blend`；`dynamic` 是 normalization 后、插值前的端点，不是 raw `exp(-log_var)`。
interpolated observation 覆盖 child observation，避免重复曲线和提醒。collector 只读，
不修改 module，也不缓存上一次 forward 的 Tensor。模块构造时会安装一次性推荐 hook；
训练中第一次未绑定 `ObservationCallback` 时在 rank 0 给出提醒。

functional helper 保持纯函数，只返回 scalar，不会因为日志配置改变计算图或 reduction。

## Temporal loss

`anytrain.loss.temporal` 提供：

- `SDRLoss`
- `global_sdr`
- `si_snr`
- `scale_invariant_signal`

`SDRLoss` 是 scalar loss module。`global_sdr` 和 `si_snr` 是底层逐样本 score helper，
保留其向量结果供下游做自定义 reduction；它们不是训练输出协议。

## Spectral loss

`LogMagnitudeLoss`、`CompressedSpectrogramLoss`、`SpectralRMSELoss` 以及 STFT/Mel 和
multi-scale 组合都返回 scalar。`STFTLoss` 和 `MelLoss` 是 `LossABC` 变换域 loss，不继承
`LossGroup`：每个实例分别持有自己的私有变换 module、命名 term `ModuleDict` 和 balancer。一次调用只对
input/target 各执行一次域变换，同一实例内的 L1/L2/log/compressed 等多个 term 复用该 spectrum，再合成一个
scalar；不同 STFT/Mel loss 或不同 scale 不共享 module 或中间 spectrum。multi-scale loss 负责 scale 间组合，
`CodecLoss` 负责 temporal/mel/STFT family 间组合。所有 term 和组合层均通过正常 module call 执行，因此各自的
scalar `OutputObservation` 会继承调用方的公共 observation context。谱损失统一使用 `torch.stft`，不再公开
transform class 或 backend 选择参数。

## Codec loss

`CodecLoss.from_preset("dac" | "dynacodec")` 组合 temporal、mel 和 STFT scalar loss；
它不解释 batch、padding、sequence lengths、训练 step 或跨 rank 加权。变长输入应由下游
在调用前裁剪或分桶，或者交给明确支持 mask/reduction 的 child loss。loss 配置不把字符串、
浮点 scale 或显式空 mapping 静默转换成其它含义。

## 边界

`loss` 不做以下事情：

- 不解释 batch schema 或 task step；
- 不隐式完成跨 shard 的 count 同步；
- 不记录日志或维护 callback 历史；
- 不把领域依赖放进默认 import；
- 不通过训练 output 自动发现中间变量。

## 测试策略

`tests/test_loss.py` 覆盖 scalar-only 返回校验、balancer 组合、frame/codebook 的有效
count reduction、空 reduction domain、packed codebook、spectral/temporal smoke、codec
preset，并检查 packed/codec 入口只经过一次 module hook，域 loss 的多个 term 共享
一次 transform，observation context 能给父子 loss 统一加前缀，公开配置入口不会执行兼容性类型转换，
uncertainty 的 `none` / `detached_sum` 数值和 surrogate gradient 符合公式，且 checkpoint 不会跨 loss
名称顺序或 normalization 模式静默恢复；interpolated balancer 覆盖 objective/weights 插值、显式 blend、
端点 observation、单次 dynamic 权重计算和 checkpoint config 校验。
Observation callback 的归约、EMA、distributed schema 和 checkpoint 行为由
observation/lightning 测试覆盖。
