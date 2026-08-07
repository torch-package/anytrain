# Stable Codec

`anytrain.audio.codec.stable_codec` 是对 Stability AI `stable-codec` 包的 optional
薄集成层。它不复制模型源码，不进入 core import；只有调用
`StableCodec.from_pretrained()` 或 `from_config()` 时才导入上游
`stable_codec`。

## 安装

官方安装方式：

```bash
python -m pip install stable-codec
python -m pip install -U flash-attn --no-build-isolation
```

当前官方 `stable-codec==0.1.2` 的 `setup.py` 硬性依赖 `torch==2.4` 和
`torchaudio==2.4`。`anytrain` 的核心下限也是 `torch>=2.4`，因此元数据不再冲突；但
Stable Codec 还要求特定 CUDA、FlashAttention 和旧构建链，所以暂不放进自动安装的 extra。
需要使用时应先创建匹配上游约束的独立环境，再安装 `anytrain` 和 Stable Codec。

为了避免 pip 重新解析已固定的 Torch 栈，可以先固定下面的 Stable Codec 依赖，再以
`--no-deps` 安装 `anytrain`，或从 checkout 暴露 wrapper 源码：

```bash
export PYTHONPATH=/path/to/anytrain/src:$PYTHONPATH
```

这个桥接环境只用于 `anytrain.audio.codec.stable_codec` 推理和离线生成 codes/latents；其它 anytrain
模块仍需满足各自依赖和运行环境约束。

Stable Codec 官方 README 还说明当前模型依赖 FlashAttention，不建议 CPU 推理。

在 `121` 上已验证 `stable-codec==0.1.2` 可以在独立 `py39` 环境安装并导入：
`python==3.9.25`、`torch==2.4.0+cu121`、`torchaudio==2.4.0+cu121`。安装时
`pypesq==1.2.4` 需要旧构建链，先安装 `numpy==1.23.5`、`Cython`、
`pip==24.0`、`setuptools==59.8.0`，再执行：

```bash
CFLAGS="-fcommon" python -m pip install --no-build-isolation pypesq==1.2.4
CFLAGS="-fcommon" python -m pip install stable-codec==0.1.2
```

复旦环境的完整记录见 `docs/codec-envs.md`。

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
