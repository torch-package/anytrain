# LongCat Audio Codec

`anytrain.audio.codec.longcat` 是对
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
export ANYTRAIN_HOME=${STATIC_HOME}/.anytrain
```

下载后的 LongCat 权重在 `$HF_HOME/longcat-audio-codec/ckpts`，生成的 patched
config 在 `$HF_HOME/longcat-audio-codec/configs`。

## 使用

```python
from anytrain.audio.codec.longcat import DEFAULT_LONGCAT_REVISION, LongCat

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

from anytrain.audio.detokenizer import load_audio_detokenizer
from anytrain.audio.tokenizer import load_audio_tokenizer

tokenizer = load_audio_tokenizer("longcat", device="cuda")
detokenizer = load_audio_detokenizer("longcat", device="cuda")
named_codes = tokenizer.tokenize(audio, sample_rate=16000)
audio_from_codes = detokenizer.detokenize(named_codes)
```

## Debug Boundary

Set `ANYTRAIN_LONGCAT_DEBUG_LOG` to a JSONL path for an explicit performance
probe. Each `LongCat.encode()` call appends one boundary record with the input
shape and audio seconds, encoder output shapes, synchronized elapsed time, CUDA
current/peak memory when running on CUDA, and an error status if the encoder
fails. The switch is disabled by default and records no waveform or token
values.

The diagnostic implementation lives in `_trace.py`. CUDA probes synchronize the
device and reset peak memory statistics at the start of encoding, so peak values
describe that measured call and affect the device's subsequent peak counters.
`_loading.py` owns optional upstream imports and temporarily sets/restores the
checkpoint environment variable, including when loading fails. `codec.py` keeps
the tensor, length, and codebook contracts.

主离散接口遵循统一 codec 契约：`audio` 是 `[batch, channels, time]`，`codes` 是
`[batch, frame, codebook]`。4-codebook decoder 的 `codebook_sizes` 是
`(8192, 8100, 8100, 8100)`；2-codebook decoder 是 `(8192, 8100)`。上游每个 acoustic
codebook 由两个 size-90 factor codebook 组成，wrapper 对外将其展平为 `90**2 == 8100` 个
local id，并在调用上游 decoder 时还原。调用方不需要拆分 semantic/acoustic 或 factor ids。

LongCat preset 把四个单码本解释为 `semantic`、`rvq_0`、`rvq_1`、`rvq_2`；每个 stream
都是 `[batch, frame]`。`encode()` / `decode()` 是 concrete backend 内部保留的拼接 tensor
接口，公共调用使用 tokenizer / detokenizer 的 named streams。

`codec.sample_rate` 是所选 decoder 的输出采样率。encoder 内部固定使用 16 kHz，
`encode(audio, sample_rate)` 会把输入采样率交给上游预处理逻辑。

对 padded batch 保留每条音频的有效 frame 数时，使用 `PaddedAudioTokenizer.tokenize_padded()`。
LongCat 的 padded 编码要求输入已经预采样到 encoder 的 16 kHz；返回的
`PaddedAudioCodes.lengths` 按 `semantic` 和各个 `rvq_*` stream 提供相同的可用 frame 数。
当 semantic encoder 多出一个尾部 frame 时，acoustic length 作为可训练的对齐边界。

`codes_to_features(stream, codes)` 对单个 named stream 返回 `[batch, time, dim]` 连续特征；
`decode_features()` 组合 semantic 与 RVQ feature streams。下游 DiT 或 flow sampler 直接接这个
feature 边界，不依赖 concrete backend 的 factor codebook 布局。

需要直接预测 LongCat factor embedding 的下游可以使用 concrete `LongCat` 的公开能力：
`acoustic_factor_codebooks` / `acoustic_factor_sizes` 暴露选定 decoder 的 factor 契约，
`project_acoustic_factor_features()`、`quantize_acoustic_residual()` 和
`native_acoustic_features()` 分别负责投影、残差量化和原生 latent 转换。上游 decoder 与
`acoustic_quantizer.quantizers` 的对象结构仍由 AnyTrain adapter 独占，下游不得访问这些私有字段。

默认从 Hugging Face 仓库 `meituan-longcat/LongCat-Audio-Codec` 下载：

- `LongCatAudioCodec_encoder.pt`
- `LongCatAudioCodec_encoder_cmvn.npy`
- `LongCatAudioCodec_decoder_16k_4codebooks.pt`
- `LongCatAudioCodec_decoder_24k_2codebooks.pt`
- `LongCatAudioCodec_decoder_24k_4codebooks.pt`

`local_files_only=True` 可以在离线环境中只使用已有缓存。
`decoders` 只会准备请求的 decoder；encoder 和 encoder cmvn 始终会准备，
缺失的请求项会按需从 Hugging Face 下载。
