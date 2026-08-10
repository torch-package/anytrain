# Stable Codec

`anytrain.audio.codec.stable_codec` 是对 Stability AI `stable-codec` 包的 optional
薄集成层。它不复制模型源码，不进入 core import；只有调用
`StableCodec.from_pretrained()` 或 `from_config()` 时才导入上游
`stable_codec`。

## 安装

`anytrain` 要求 Python `>=3.12`。当前官方 `stable-codec==0.1.2` 仍依赖固定的
`torch==2.4`、`torchaudio==2.4`、特定 CUDA、FlashAttention 和旧构建链；其依赖链在
Python 3.12 环境中尚未形成受支持的安装组合。因此 Stable Codec 暂不提供 extra，wrapper
源码仅保留集成边界，不应通过较低 Python 版本或 `PYTHONPATH` bridge 绕过 anytrain 的包约束。

历史环境和 API smoke 记录保留在 `docs/codec-envs.md`，只作为上游行为证据，不代表当前支持
策略。待上游依赖链支持 Python 3.12 后，再为该 wrapper 恢复可执行的安装说明。Stable Codec
官方 README 还说明当前模型依赖 FlashAttention，不建议 CPU 推理。

## 使用

```python
from anytrain.audio.codec.stable_codec import StableCodec

codec = StableCodec.from_pretrained(
    version="speech-16k",
    device="cuda",
)

codes = codec.encode(audio, sample_rate=16000)
audio_out = codec.decode(codes)

latents, codes = codec.encode_latents(audio, sample_rate=16000)
```

`audio` 需要是 `[batch, 1, time]` 的 16 kHz 单声道 waveform。Stable Codec 没有
UniCodec 的 `domain` 参数；默认模型是 `stabilityai/stable-codec-speech-16k`。
官方 speech 模型通过 `downsampling_ratio=640` 暴露帧步长，因此 wrapper 的
`frame_rate` 为 `16000 / 640 = 25 Hz`。

主路径按统一 codec 契约设计：`encode()` 返回 `[batch, frame, codebook]` codes，
`decode()` 接收同样形状的 codes
并重建 waveform。上游实现同时返回 pre-bottleneck continuous latents；需要直接操作这个
边界时使用 `encode_latents()`。

wrapper 默认启用单码本 `1x46656_400bps` posthoc bottleneck，默认
`codebook_sizes` 为 `(46656,)`。需要使用训练期 native FSQ 时显式传入
`posthoc_bottleneck=None`；native speech 模型使用六维、每维 17 级的训练期 FSQ，
因此 native `codebook_sizes` 是 `(17^6,)`，即 `(24137569,)`，不是 `(46656,)`。
wrapper 从上游 `model.bottleneck.quantizer` 读取真实 `codebook_size` 和
`num_codebooks`，不为本地 config 猜测码本大小。

`121` 上确认的上游 `0.1.2` 签名与 wrapper 对齐：

- `StableCodec(pretrained_model=..., device=...)`
- `StableCodec(model_config_path=..., ckpt_path=..., device=...)`
- `encode(audio, posthoc_bottleneck=False, normalize=True, **kwargs)`
- `decode(tokens, posthoc_bottleneck=False, **kwargs)`
- `set_posthoc_bottleneck(stages)`

## Posthoc Bottleneck

上游推荐 posthoc FSQ bottleneck。wrapper 默认使用 `1x46656_400bps`，也可以在初始化时选择其他 preset：

```python
codec = StableCodec.from_pretrained(
    device="cuda",
    posthoc_bottleneck="2x15625_700bps",
)
```

也可以先加载模型，再设置：

```python
codec.set_posthoc_bottleneck("2x15625_700bps")
codes = codec.encode(audio, sample_rate=16000)
audio_out = codec.decode(codes)
```

支持的上游 preset：

- `"1x46656_400bps"`
- `"2x15625_700bps"`
- `"4x729_1000bps"`

`stable-codec==0.1.2` 的 posthoc encode 为每个 stage 返回一个 `[batch, frame, 1]`
Tensor 的 list。wrapper 会沿最后一维拼接为公共 `[batch, frame, codebook]` Tensor；
decode 时再拆回上游所需的 list。`46656` 仅是 `1x46656_400bps` preset 的码本大小，
需要把 Stable Codec tokens 交给语言模型时，产物配置仍应记录实际使用的 preset。

如果需要本地 config 和 checkpoint：

```python
codec = StableCodec.from_config(
    "config.json",
    ckpt_path="model.ckpt",
    device="cuda",
)
```
