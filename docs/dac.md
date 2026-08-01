# Descript Audio Codec

`anytrain.codec.dac` 是对 Descript Audio Codec 的 optional 薄集成。它不复制模型源码，
也不在 import `anytrain.codec` 时导入上游 `dac` 包；只有加载 checkpoint 时才需要安装
`descript-audio-codec`。

## 安装

```bash
python -m pip install "descript-audio-codec>=1.0" "torchcodec>=0.2" "torchaudio>=2.0"
```

`dac` extra 提供上游模型，`audio` extra 提供输入采样率不一致时使用的 torchaudio
重采样。输入已经是模型采样率时不需要 torchaudio。

## 使用

```python
from anytrain.codec.dac import DAC

codec = DAC.from_pretrained(
    model_type="44khz",
    model_bitrate="8kbps",
    cache_dir="/path/to/model-cache/dac",
    device="cuda",
)

codes = codec.encode(audio, sample_rate=48000)
audio_out = codec.decode(codes)
```

`audio` 使用 `[batch, 1, time]`，`codes` 使用统一的
`[batch, frame, codebook]`。DAC 上游内部使用 `[batch, codebook, frame]`，wrapper
负责轴转换。`decode()` 先通过 `quantizer.from_codes()` 恢复量化特征，再调用上游 decoder。

默认加载官方 `44khz` / `8kbps` / `latest` checkpoint。官方可用组合为：

- `16khz` / `8kbps`
- `24khz` / `8kbps`
- `44khz` / `8kbps`
- `44khz` / `16kbps`

需要低码率推理时，在构造阶段固定量化器数量：

```python
codec = DAC.from_pretrained(
    model_type="44khz",
    n_quantizers=4,
    cache_dir="/path/to/model-cache/dac",
)
```

此时 `codebook_sizes` 和 `decode()` 都按前 4 个有序码本工作。不能向该实例传入其它数量
或重排后的码本。

## Checkpoint 与缓存

自动下载不会调用 DAC 上游硬编码到 `~/.cache/descript/dac` 的 downloader。默认缓存目录是
`ANYTRAIN_HOME/dac`，也可以通过 `cache_dir` 显式指定。共享机器上应优先显式传入稳定模型
目录，避免权重写入用户 home 或根分区。
远程下载默认要求 wrapper 维护对应 checkpoint 的 SHA-256 元数据；没有校验元数据时会拒绝
下载。确实需要沿用旧的不校验下载路径时，显式传入 `allow_unsafe_download=True`。

已有本地 checkpoint 时直接加载：

```python
codec = DAC.from_checkpoint(
    "/path/to/weights_44khz_16kbps_1.0.0.pth",
    device="cuda",
)
```

`local_files_only=True` 会在缓存缺失时直接抛出 `FileNotFoundError`；
`force_download=True` 会重新下载并原子替换目标 checkpoint。

DAC 会把输入补齐到模型 hop length，因此仅凭 codes 解码时，输出末尾可能包含少量 padding。
下游如果需要恢复原始精确长度，应同时保存重采样后的 waveform 长度并在 decode 后裁剪。
