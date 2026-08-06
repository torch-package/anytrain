# LongCat Audio Codec

`anytrain.codec.longcat` 是对
`meituan-longcat/LongCat-Audio-Codec` 的 optional 集成层。它不进入 core import，
只在用户显式安装 `longcat` extra 并调用相关 API 时使用。

## 安装

LongCat 源码维护在可安装 fork：

```bash
python -m pip install -e ../LongCat-Audio-Codec
```

fork 推送后，下游可以直接安装 anytrain extra：

```bash
python -m pip install "anytrain[longcat]"
```

`longcat` extra 依赖：

- `longcat-audio-codec @ git+https://github.com/rookiejune/LongCat-Audio-Codec.git`
- `huggingface-hub`
- `PyYAML`

## 缓存路径

checkpoint 自动下载到缓存目录。路径优先级：

1. 显式传入 `cache_dir=...`
2. `HF_HOME/longcat-audio-codec`
3. 如果 `HF_HOME` 未设置，anytrain 会设置
   `HF_HOME=${ANYTRAIN_HOME:-~/.anytrain}/huggingface`，再使用
   `$HF_HOME/longcat-audio-codec`

远程下载默认要求显式传入 Hugging Face `revision=<commit-sha>`；若需要沿用旧的浮动分支
下载行为，必须显式传入 `allow_unpinned_revision=True`。
内置 `longcat` preset 使用 `DEFAULT_LONGCAT_REVISION` 固定权重版本。

远程服务器上推荐：

```bash
export ANYTRAIN_HOME=/mnt/pami202/zhuyin/.anytrain
```

下载后的 LongCat 权重在 `$HF_HOME/longcat-audio-codec/ckpts`，生成的 patched
config 在 `$HF_HOME/longcat-audio-codec/configs`。

## 使用

```python
from anytrain.codec.longcat import DEFAULT_LONGCAT_REVISION, LongCat

codec = LongCat.from_pretrained(
    device="cuda",
    decoder="24k_4codebooks",
    revision=DEFAULT_LONGCAT_REVISION,
)

codes = codec.encode(
    audio,
    sample_rate=16000,
)
audio_24k = codec.decode(codes)

units = codec.tokenize(audio, sample_rate=16000)
audio_from_units = codec.detokenize(units)

acoustic_features = codec.acoustic_codes_to_features(
    codes[..., 1:],
)
audio_from_features = codec.decode_features(
    codes[..., :1],
    acoustic_features,
)
```

主离散接口遵循统一 codec 契约：`audio` 是 `[batch, channels, time]`，`codes` 是
`[batch, frame, codebook]`。4-codebook decoder 的 `codebook_sizes` 是
`(8192, 8100, 8100, 8100)`；2-codebook decoder 是 `(8192, 8100)`。上游每个 acoustic
codebook 由两个 size-90 factor codebook 组成，wrapper 对外将其展平为 `90**2 == 8100` 个
local id，并在调用上游 decoder 时还原。调用方不需要拆分 semantic/acoustic 或 factor ids。

LongCat 同时满足 `FrameCodec` 和 `SemanticAcousticCodec`。structured 路径中，
`units.semantic` 是 `[batch, frame, 1]`，`units.acoustic` 是
`[batch, frame, acoustic_codebook]`，`acoustic_layout` 为 `FRAME_ALIGNED`。
`encode()` / `decode()` 保留原有拼接 tensor 契约；需要独立 semantic/acoustic 边界时使用
`tokenize()` / `detokenize()`。

`codec.sample_rate` 是所选 decoder 的输出采样率。encoder 内部固定使用 16 kHz，
`encode(audio, sample_rate)` 会把输入采样率交给上游预处理逻辑。

`acoustic_codes_to_features()` 是 LongCat 专用扩展，输入仍遵循统一的
`[batch, time, codebook]` 契约，并显式调用 decoder 的 acoustic dequantizer，对外返回
`[batch, time, dim]` 的连续 acoustic features。`decode_features()` 的 semantic codes
同样保留 codebook 轴；LongCat 当前要求 `[batch, time, 1]`。wrapper 负责转换成上游
decoder 的内部 shape，调用方不做转置或 squeeze。下游 DiT 或 flow sampler 应该接这个
feature 边界；只有原始 codec roundtrip 才直接使用离散 `acoustic_codes`。

默认从 Hugging Face 仓库 `meituan-longcat/LongCat-Audio-Codec` 下载：

- `LongCatAudioCodec_encoder.pt`
- `LongCatAudioCodec_encoder_cmvn.npy`
- `LongCatAudioCodec_decoder_16k_4codebooks.pt`
- `LongCatAudioCodec_decoder_24k_2codebooks.pt`
- `LongCatAudioCodec_decoder_24k_4codebooks.pt`

`local_files_only=True` 可以在离线环境中只使用已有缓存。
`decoders` 只会准备请求的 decoder；encoder 和 encoder cmvn 始终会准备，
缺失的请求项会按需从 Hugging Face 下载。
