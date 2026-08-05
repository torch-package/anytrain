# Codec

`anytrain.codec` 为 optional audio codec wrapper 提供轻量 runtime protocol 和加载前的
capability 分类。anytrain 区分三种 backend 能力：完整 frame-code codec、显式拆分且仍表达
声学时间结构的 semantic/acoustic codec，以及语义时间轴外另带全局 unit 的 semantic/global
codec。semantic-only waveform decoder 属于下游 `semantic-acoustic-codec` runtime，不属于 backend。

## Protocol

所有 codec 共享最小身份边界：

```python
from typing_extensions import Protocol
from torch import Tensor

class CodecBackend(Protocol):
    name: str
    sample_rate: int
    frame_rate: float


class FrameCodec(CodecBackend, Protocol):
    codebook_sizes: tuple[int, ...]

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...
    def decode(self, codes: Tensor) -> Tensor: ...
```

- `audio` 使用 `[batch, channels, time]`。
- `encode()` 接收输入 waveform 的真实采样率，wrapper 在需要时内部重采样。
- `codes` 是有符号整数 Tensor，统一使用 `[batch, frame, codebook]`。
- `sample_rate` 表示 `decode()` 输出 waveform 的采样率。
- `codebook_sizes[k]` 是第 `k` 个码本的 local id 数量，且每个值必须是正整数。
- `decode()` 接收 codec 实例配置的完整、有序 K 个码本；码本轴不能重排或用任意子集冒充。
- `decode()` 前，`codes[..., k]` 的每个值必须满足
  `0 <= codes[..., k] < codebook_sizes[k]`。
- 当前 concrete wrapper 都是 `nn.Module`，公开只读 `device` 属性；调用 `.to()`、
  `.cuda()` 或 `.cpu()` 时，backend 子模块和该设备属性会一起迁移。设备标记不写入
  `state_dict()`。直接构造 wrapper 时也会把 backend 统一迁移到指定设备；普通
  `load_state_dict()` 保持设备不变。wrapper 不支持可能替换 tensor device 的
  `load_state_dict(assign=True)`，会明确抛出 `ValueError`。

显式拆分 semantic 与 acoustic 单元的 codec 另外满足：

```python
class SemanticAcousticCodec(CodecBackend, Protocol):
    semantic_codebook_sizes: tuple[int, ...]
    acoustic_codebook_sizes: tuple[int, ...]
    acoustic_layout: AcousticLayout
    semantic_codebook: Tensor
    semantic_frame_rate: float
    acoustic_feature_dim: int
    acoustic_unit_length: int | None

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes: ...
    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...
    def decode_features(self, semantic_codes: Tensor, acoustic_features: Tensor) -> Tensor: ...
```

`SemanticAcousticCodes.semantic` 统一使用 `[batch, time, codebook]`；`acoustic` 使用
`[batch, acoustic_unit, codebook]`。`FRAME_ALIGNED` 要求两个 unit 轴对齐，
`FIXED_LENGTH` 表示 acoustic unit 数由 codec 固定，不随 semantic 帧数变化。

需要把 acoustic code 转成连续训练特征时，使用
`anytrain.codec.masked_acoustic_features(codec, acoustic_codes, mask)`。该 helper 会先把
padding code 置零，再调用 backend，并对输出特征重新应用 unit mask，避免 padding 泄漏到
下游 loss；已经由上游 batch 契约验证过的内部热路径可以传 `validate=False`。

semantic/acoustic codec 的 feature training boundary 把 feature tensor 保持为
`[batch, acoustic_unit, feature_dim]`；`acoustic_unit_length=None` 只适用于可变长度的
`FRAME_ALIGNED` codec。

语义时间轴外另带全局 unit 的 codec 使用独立契约：

```python
class SemanticGlobalCodec(CodecBackend, Protocol):
    semantic_codebook_sizes: tuple[int, ...]
    global_codebook_sizes: tuple[int, ...]
    semantic_codebook: Tensor
    semantic_frame_rate: float
    global_feature_dim: int
    global_unit_length: int

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticGlobalCodes: ...
    def detokenize(self, codes: SemanticGlobalCodes) -> Tensor: ...

    def global_codes_to_features(self, global_codes: Tensor) -> Tensor: ...
    def decode_features(self, semantic_codes: Tensor, global_features: Tensor) -> Tensor: ...
```

`SemanticGlobalCodes.semantic` 使用 `[batch, time, codebook]`，`global_codes` 使用
`[batch, global_unit, codebook]`。`global_unit_length` 是严格正整数；global axis 与 semantic
time axis 独立，不使用 `AcousticLayout`，也不能广播成逐帧声学 stream。BiCodec 的 32 个 speaker
slot 属于这个契约。

需要在不持有 backend 权重的层之间传递结构时，分别使用 `SemanticAcousticCodecSpec` 或
`SemanticGlobalCodecSpec`。`semantic_acoustic_spec(codec)` 与 `semantic_global_spec(codec)` 从已实例化
backend 读取并严格校验对应的 rate、codebook、feature dim 和 unit 结构；spec 不包含 backend 名称、
checkpoint、加载参数或实际 codebook Tensor，因此不能替代 codec 实例和 registry。

这些 protocol 是可重叠的 capability，不是 concrete class 继承树：

| Codec | registry name | `FrameCodec` | `SemanticAcousticCodec` | `SemanticGlobalCodec` | feature target |
|---|---|---:|---:|---:|---|
| DAC | `dac` | 是 | 否 | 否 | - |
| Stable Codec | `stable_codec` | 是 | 否 | 否 | - |
| UniCodec | `unicodec` | 是 | 否 | 否 | - |
| LongCat | `longcat` | 是 | 是 | 否 | frame-aligned acoustic |
| BiCodec | `bicodec` | 否 | 否 | 是 | independent global units |

Stable Codec 的 latent 可以被语义目标增强，DAC 的早期 RVQ codebook 也可能包含较粗信息；但它们
没有公开独立 semantic stream，因此不声明 structured capability。LongCat 同时保留完整 frame codes
和 semantic/acoustic 接口。BiCodec 的 semantic/global unit 不能无损拼成一个共享 frame 轴，因此只
声明 `SEMANTIC_GLOBAL`；它不会通过 semantic/acoustic loader 暴露。

## Registry

配置只保存 registry name。加载前可以查询 capability；调用方使用对应的窄 loader，传入不匹配的
codec 时会在下载权重前明确报错：

```python
from anytrain.codec import CodecCapability, capabilities, load_frame, load_semantic_global

assert CodecCapability.SEMANTIC_GLOBAL in capabilities("bicodec")
codec = load_semantic_global("bicodec", device="cuda")
frame_codec = load_frame("longcat", device="cuda")
```

registry 只保存稳定名称、capability 和默认 lazy loader，不保存 sample rate、码本大小或 layout 的
副本；这些信息由实际 codec 实例提供。非默认 decoder、checkpoint 或 cache 选项仍从具体 backend
子模块构造。

Stable Codec 的 posthoc bottleneck、LongCat decoder、UniCodec domain / bandwidth 和 DAC
`n_quantizers` 在构造 codec 时固定，不进入公共 encode/decode 参数。

完整 frame-code 路径由 `load_frame()` 提供；semantic/acoustic 与 semantic/global 路径分别由
`load_semantic_acoustic()` 和 `load_semantic_global()` 提供。LongCat 同时支持 frame 和
semantic/acoustic；BiCodec 只支持 semantic/global；DAC、Stable Codec 和 UniCodec 只支持 frame codes。

具体 wrapper：

- [Descript Audio Codec](dac.md)
- [Spark-TTS BiCodec](bicodec.md)
- [LongCat Audio Codec](longcat.md)
- [Stable Codec](stable-codec.md)
- [UniCodec](unicodec.md)
