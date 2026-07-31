# `anytrain.loss` Design

## 定位

`anytrain.loss` 提供下游 LightningModule 可直接组合使用的通用 loss 组件。core loss 不绑定具体任务领域；audio/text/speech 等领域 loss 应放在 optional 子模块。GAN adversarial training 属于 `anytrain.framework.gan`，不放在 `loss` 下。

## 当前结构

源码结构：

```text
src/anytrain/loss/
  __init__.py
  abc.py
  balancer.py
  codebook.py
  frame.py
  group.py
  temporal/
    __init__.py
    single.py
  spectral/
    __init__.py
    single.py
    transform.py
    group.py
  codec.py
  task/
    __init__.py  # compatibility re-export
```

当前公开导出：

- `LossABC`
- `LossResult`
- `LossDetails`
- `LossDetailValue`
- `LossBalancerABC`
- `LossTensorDict`
- `MeanLossBalancer`
- `FixedWeightLossBalancer`
- `UncertaintyLossBalancer`
- `LossGroup`
- `LossItem`
- `MaskedFrameMSELoss` / `masked_frame_mse_loss`
- `MaskedCosineAlignmentLoss` / `masked_cosine_alignment_loss`
- `MaskedCodebookCrossEntropyLoss`

音频相关 loss 通过子模块导出：

- `anytrain.loss.temporal.SDRLoss`
- `anytrain.loss.spectral.LogMagnitudeLoss`
- `anytrain.loss.spectral.CompressedSpectrogramLoss`
- `anytrain.loss.spectral.SpectralRMSELoss`
- `anytrain.loss.spectral.STFTLoss`
- `anytrain.loss.spectral.MelLoss`
- `anytrain.loss.spectral.MultiScaleSTFTLoss`
- `anytrain.loss.spectral.MultiScaleMelLoss`
- `anytrain.loss.spectral.STFTTransform`
- `anytrain.loss.spectral.MelSpectrogramTransform`
- `anytrain.loss.codec.CodecLoss`
- `anytrain.loss.codec.CodecLossPreset`

## Loss 协议

`LossABC` 是继承 `torch.nn.Module` 的 loss 抽象基类。子类实现：

```python
def compute_loss(self, *args, **kwargs) -> LossResult:
    ...
```

`LossResult` 可以是：

```python
loss
loss, details
```

协议约束：

- `loss` 必须是 0-d `torch.Tensor`，可以直接用于 `backward()`。
- `details` 是可选的 mapping，只用于监控和日志。
- `details` 的 key 必须是非空字符串，普通子 loss 中不能包含 `/`。
- `details` 的 value 必须是 float 或 0-d `torch.Tensor`。
- 返回给调用方的 `details` tensor 会被 detach，避免把日志路径接进反传图。

`LossItem` 保存按样本/行聚合的 loss 与可选 details。`MaskedFrameMSELoss` 和
`MaskedCosineAlignmentLoss` 对 `[B, F, D]` 表示执行 frame mask 后的 MSE 或 cosine alignment；
`MaskedCodebookCrossEntropyLoss` 接收每个 codebook 一个 `[B, F, vocab_q]` logits，支持不同
vocab size、frame mask、逐 codebook CE 和可选 `codebook_N_top1` 监控。

## 基础 Loss

根目录不再保留 `single.py`。基础 L1/MSE/CE 等通用目标直接使用 `torch.nn`，需要 `details` 或额外校验时由下游写一个继承 `LossABC` 的小模块显式聚合成 scalar 主 loss。

## 组合 Loss

`LossGroup` 接收 loss name 到 loss module 的 mapping，并返回：

```python
total, details = loss_fn(*args, **kwargs)
```

行为约定：

- `losses` 不能为空。
- `losses` 内部使用 `nn.ModuleDict` 注册。
- `balancer` 是可选的 `LossBalancerABC`，默认使用 `MeanLossBalancer`。
- `MeanLossBalancer` 对所有子 loss 求平均，得到可反传的 total。
- `FixedWeightLossBalancer` 用显式 mapping 做固定权重组合。
- `UncertaintyLossBalancer` 从 deepaudio 的 uncertainty balancer 迁入，按显式 `loss_names` 顺序维护可训练 `log_var`。
- balancer 可以返回 scalar Tensor 或 `(scalar Tensor, details)`。
- 每个子 loss 以同一组 `*args, **kwargs` 调用。
- 子 loss 可以返回 scalar Tensor 或 `(scalar Tensor, details)`。
- `details[name]` 记录未加权 loss 的 detached 值。
- 子 loss 的 details 会用 `"{loss_name}/{detail_key}"` 前缀合并。
- balancer 的 details 会用 `"balancer/{detail_key}"` 前缀合并。
- total 只通过返回值中的第一个元素提供，不重复写入 details。

`LossGroup` 不直接提供 `loss_weights` 参数。固定权重、动态权重或领域特定平衡策略都应实现为 balancer；例如 `deepaudio.loss` 可以提供自己的 audio loss balancer。

## Temporal Loss

`anytrain.loss.temporal` 放 waveform 域 loss。

当前提供：

- `SDRLoss`：负 SDR / SI-SNR，输入形状通常是 `(batch, channels, time)`。
- `global_sdr`
- `si_snr`
- `scale_invariant_signal`

## Spectral Loss

`anytrain.loss.spectral.single` 放已经在谱域或 mel 域的单项 loss：

- `LogMagnitudeLoss`
- `CompressedSpectrogramLoss`
- `SpectralRMSELoss`

`anytrain.loss.spectral.group` 放负责变换和组合的 loss：

- `STFTLoss`
- `MelLoss`
- `MultiScaleSTFTLoss`
- `MultiScaleMelLoss`

`STFTLoss` 和 `MelLoss` 先把 waveform 转成表示，再调用内部 `LossGroup`。`MultiScaleSTFTLoss` 和 `MultiScaleMelLoss` 再用多个尺度的 `STFTLoss` / `MelLoss` 组成多尺度目标。

`CodecLoss` 在单次 `forward` 内为重叠的 multi-scale mel/STFT 配置共享原始 STFT，
仅当后端和 STFT 参数完全一致时复用；缓存不会跨 forward 保留。torchaudio 后端的
mel 路径使用相同的原始 `Spectrogram` 和已有 `MelScale`，与未缓存路径保持数值等价。

谱变换支持 `backend="auto" | "torchaudio" | "torch"`：

- `auto` 默认优先使用 `torchaudio.transforms`。
- 环境没有 `torchaudio` 时，`auto` 回退到 torch-only 实现。
- `torchaudio` 会强制要求安装 `torchaudio`。
- `torch` 使用 `torch.stft` 和本地 mel filter bank fallback。

Mel 变换还支持 `mel_scale="htk" | "slaney"`，默认保持 torchaudio/deepaudio 的 `"htk"`。

## Codec Loss

`anytrain.loss.codec` 放 codec reconstruction preset 组合，但不解释 batch 或训练 step。
下游应直接从 `anytrain.loss.codec` 导入；不提供 `anytrain.loss.task` 这类 task 级快捷别名。

当前提供 `CodecLoss.from_preset(...)`：

- `CodecLossPreset.DAC` / `"dac"`：从 deepaudio 迁入 DAC reconstruction preset，只包含 multi-scale mel loss，默认尺度为 `n_fft=(2048, 1024, 512, 256, 128, 64, 32)`、`n_mels=(320, 160, 80, 40, 20, 10, 5)`。
- `CodecLossPreset.DYNACODEC` / `"dynacodec"`：从 deepaudio 迁入 DynaCodec reconstruction preset，包含 `si_sdr`、`multi_mel` 和 `multi_stft`，默认固定权重为 `1.0 / 10.0 / 10.0`。

示例：

```python
from anytrain.loss.codec import CodecLoss

loss_fn = CodecLoss.from_preset("dynacodec")
```

变长音频可以传 `lengths`，`CodecLoss` 会逐样本裁剪有效时间后求 batch 平均，避免
padding 污染重建 loss：

```python
loss, details = loss_fn(reconstruction, target, lengths=batch.lengths)
```

`lengths` 必须是一维整数 Tensor，或由 index-compatible integer 组成的 sequence；bool、浮点、
复数和字符串会直接抛出 `TypeError`，不会通过 `int()` 截断或转换。dict、generator、bytes
等非 sequence 或文本/二进制容器也不作为 batch lengths 接受。它必须为每个 batch item
提供一个正数，且不能超过输入的 time 维。传入 `lengths` 时，`reconstruction` 和 `target`
必须同 shape，并至少具有 `(batch, channels, time)` 三个维度。

## 当前限制

当前 core 已有最小可用实现，但还没有：

- plain-config 友好的 loss 组合/工厂，用于按配置装配 `LossABC` / `LossGroup`
  和 balancer。
- deviation 等更多 balancer 策略。

这些属于后续演进，不应在下游文档里假设已经存在。

## 边界

`loss` 不做：

- 不解释 batch。
- 不决定训练 step 如何调用 loss。
- 不记录日志；logging 由下游 LightningModule 处理。
- 不把领域依赖放进默认 import。

## 测试策略

当前覆盖：

- `LossGroup` 的 total 和 detail key。
- `FixedWeightLossBalancer` 的固定权重组合。
- `SDRLoss`、spectral single loss、STFT/Mel 和 multi-scale spectral loss 的 smoke test。
- `CodecLoss` DAC / DynaCodec preset 结构、length-aware 裁剪和错误 preset 路径。

后续扩展 plain-config 装配 helper 时，应增加嵌套 details flatten、权重 device
迁移、checkpoint round-trip 和错误路径测试。
