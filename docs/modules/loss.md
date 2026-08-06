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
- `LossTensorDict`
- `MeanLossBalancer`
- `FixedWeightLossBalancer`
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
  有效 frame 求均值，最后对 codebook 求均值。`forward_packed()` 接受
  `PackedCodebookLogits(logits, labels)`，其中 logits 是每个 codebook 的 `[N, vocab]`
  Tensor，labels 是 `[N, Q]`。旧调用方可以继续附带 `row_indices`/`batch_size` packing
  metadata；这些字段只做一致性校验，不参与 scalar reduction。这个公开的 packed 入口
  仍通过 module call 执行，因此与普通入口一样触发 output observation 和推荐 hook。
- `SDRLoss`：先按样本计算 SDR/SI-SNR，再对 batch 求均值。
- `CodecLoss(lengths=...)`：按相同长度分组裁剪有效区间，使用
  `group scalar * group sample count / batch sample count` 合并，避免 padding 污染并保持
  样本等权。这个路径要求每个 child scalar 表示该等长 group 的 batch mean；内置 temporal
  和 spectral child 均满足。`reduction="sum"` 或其它不具备 batch-mean 语义的自定义 child
  不属于这个组合契约，因为 scalar 返回后无法推断其 reduction count。

frame/codebook 的有效 reduction domain 为空时，loss 会抛出 `ValueError`，不会返回
`nan` 或静默的零值。

### 分片边界

如果不同 rank、数据 shard 或 gradient-accumulation microbatch 覆盖的有效 count 不同，
对各自返回的 scalar 直接求平均并不等价于全局 count-weighted mean。调用方必须满足
以下之一：让一次 loss 调用覆盖完整 logical domain；保证各 shard 的 count 相等；或在
训练编排层用调用方已知的 count 对目标显式缩放。loss API 不返回公开 count，也不尝试
在 callback 中猜测或补做跨 rank reduction。

## Observation registry

诊断的唯一入口是 `anytrain.observation` registry；不再有 loss-specific logger callback
或 `details` mapping。

已注册 loss module 的最终输出通过
`OutputObservation("loss", ..., reduction=observation.Reduction.Mean)` 记录为 `Curve`；logging-only scalar
diagnostics 的 `ForwardEvent` 同样显式使用 `Mean`。
`LossGroup` 只声明自己的 total；嵌套 child 若有稳定诊断，则由 child 自己声明，避免
group 复制或缓存 child 的中间状态。

只存在于一次 forward 局部作用域的值通过 `ForwardEvent("diagnostics")` 发布：

- frame loss 发布 `valid_frames`，cosine/teacher 还发布 cosine 和 smooth-L1 scalar；
- codebook loss 发布每个 codebook 的 CE 与 `valid_frames`；传入 `include_top1=True`
  时额外发布每个 codebook 的 top-1 scalar。

balancer 的持久属性通过 `StateObservation("weights", ...)` 读取：固定权重和
uncertainty 权重都可看到当前值与历史曲线。collector 只读，不修改 module，也不缓存
上一次 forward 的 Tensor。模块构造时会安装一次性推荐 hook；训练中第一次未绑定
`ObservationCallback` 时在 rank 0 给出提醒。

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
multi-scale 组合都返回 scalar。变换支持 `backend="auto" | "torchaudio" | "torch"`，
并保持现有 cache 与 mel-scale 语义。cache 通过内部 keyword 沿普通 module call 传递，
不会绕过嵌套 STFT/Mel loss 的 output observation 或推荐 hook。

## Codec loss

`CodecLoss.from_preset("dac" | "dynacodec")` 组合 temporal、mel 和 STFT scalar loss；
它不解释 batch 或训练 step。传入 `lengths` 时，输入和 target 必须同 shape，lengths 为
每个 batch item 的正整数且不超过 time 维。

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
preset 与 length-aware 路径，并检查 packed/cache-aware 入口只经过一次 module hook。
Observation callback 的归约、EMA、distributed schema 和 checkpoint 行为由
observation/lightning 测试覆盖。
