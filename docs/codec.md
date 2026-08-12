# Audio Tokenizer 与 Detokenizer

`anytrain.audio` 对外只提供两个方向性边界：waveform 到 named code streams 的
`AudioTokenizer`，以及 named code streams 到 features 再到 waveform 的
`AudioDetokenizer`。调用方不依赖 concrete codec backend，也不需要知道 backend 的原生
tensor 布局。

## Code spec

`AudioCodeSpec` 静态描述一个 preset 的完整码流清单：

```python
from anytrain.audio.codec import audio_spec

spec = audio_spec("longcat")
assert tuple(stream.name for stream in spec.streams) == (
    "semantic",
    "rvq_0",
    "rvq_1",
    "rvq_2",
)
assert spec.stream("semantic").vocab_size == 8192
assert spec.stream("semantic").frame_rate == 50.0
```

一个 `AudioCodeStreamSpec` 只表示一个命名码本。RVQ 逐级写成 `rvq_0`、`rvq_1` 等；
随时间变化的 stream 声明 `frame_rate`，固定长度 stream 声明 `fixed_units`，二者互斥。
例如 BiCodec `global` 是固定 32 units，不是 32 Hz。`select()` 和 `exclude()` 可以构造
子清单，但 tokenizer 必须返回其自身 `code_spec.streams` 中的全部码流。


## Tokenizer

```python
from anytrain.audio.tokenizer import load_audio_tokenizer

tokenizer = load_audio_tokenizer("longcat", device="cuda")
codes = tokenizer.tokenize(audio, sample_rate=48_000)
semantic = codes["semantic"]  # [batch, time]
rvq_0 = codes["rvq_0"]        # [batch, time]
```

`tokenize()` 返回 `Mapping[str, Tensor]`，key 必须与 `code_spec.streams` 的 name 完全一致。每个 value
都是 `[batch, unit]` 的 signed integer ID；ABC 统一校验 stream inventory、batch、dtype 和
vocabulary range。

支持变长真 batch 的 tokenizer 还实现 `PaddedAudioTokenizer`：

```python
from anytrain.audio.tokenizer import PaddedAudioTokenizer

if isinstance(tokenizer, PaddedAudioTokenizer):
    encoded = tokenizer.tokenize_padded(audio, sample_rate, waveform_lengths)
    codes = encoded.codes
    valid_lengths = encoded.lengths
```

`audio` 是 `[batch, channel, padded_time]`；`waveform_lengths` 是未 padding 的输入长度。
`PaddedAudioCodes.lengths` 按 stream 返回 `[batch]` 有效 unit 数，因为 fixed/global stream
可以与 frame stream 使用不同长度。调用方只消费每条 stream 的有效前缀。

## Detokenizer

```python
from anytrain.audio.detokenizer import load_audio_detokenizer

detokenizer = load_audio_detokenizer("longcat", device="cuda")
feature = detokenizer.codes_to_features("rvq_0", codes["rvq_0"])
audio = detokenizer.detokenize(codes)
```

`AudioDetokenizer` 的基础操作是：

```python
class AudioDetokenizer(ABC):
    @property
    def code_spec(self) -> AudioCodeSpec: ...

    def codes_to_features(self, stream: str, codes: Tensor) -> Tensor: ...
    def decode_features(self, features: Mapping[str, Tensor | None]) -> Tensor: ...
```

`detokenize(codes)` 是 ABC 提供的便捷入口：按 `code_spec.streams` 对每个非空码流调用
`codes_to_features()`，再调用一次 `decode_features()`。训练代码可以直接生成 features 并跳过
离散码；是否允许某个 feature stream 为 `None` 由具体 detokenizer 决定。

每个 stream 的 feature 是 `[batch, unit, feature]`。不同 stream 可以有不同 unit 轴，例如
BiCodec 的 `semantic` 是时间轴，`global` 是固定 32 个 speaker units。调用方不得把独立 unit
轴广播成 frame 轴。

## Registry

```python
from anytrain.audio.codec import (
    audio_backend_identity,
    audio_capabilities,
    audio_planning_estimator,
    audio_spec,
    can_load_audio_detokenizer,
    can_load_audio_tokenizer,
)
```

这些查询不加载模型。`load_audio_tokenizer()` 与 `load_audio_detokenizer()` 通过 preset 的
representation adapter 返回方向对象；同一 preset、identity 和 device 可以共享内部 backend
实例。共享是 registry 的实现细节，调用方只依赖方向 ABC。

GLM-4 只声明 tokenizer 方向；LongCat、BiCodec、DAC、Stable Codec 和 UniCodec 声明已实现的
方向。具体 backend 的原生 `encode()`、`decode()` 或 feature helper 只服务 preset adapter，
不构成公共 aggregate codec 协议。

BiCodec 与 GLM-4 还为离线 batching cost 暴露静态 estimator：

```python
estimator = audio_planning_estimator("bicodec")
units = estimator.estimate(source_frames=48_000, source_rate=48_000)
sidecar_identity = estimator.identity
```

查询和估算只使用 pinned backend 的整数长度契约，不加载模型或 optional backend。`identity`
同时绑定 backend artifact/revision、估算算法版本和数值参数；`version` 可用于诊断公式版本。
BiCodec 按未 padding 的 Wav2Vec2 convolution 逐层计算，过短输入明确失败；GLM-4 按 feature hop、
convolution/pooling 和官方 context chunk 边界计算。该值只用于规划 tokenizer unit cost，不代表
真实音频时长，也不能作为 tokenization 输出正确性的验收证据。没有已验证静态契约的 preset 会在
`audio_planning_estimator(name)` 明确失败。

`anytrain.audio.bpe._frame.FrameCodec` 是 BPE 序列化使用的本地 frame 值对象，与神经网络
audio codec API 无关。
