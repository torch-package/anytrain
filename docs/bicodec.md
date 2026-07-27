# Spark-TTS BiCodec

`anytrain.codec.bicodec` 是对 SparkAudio Spark-TTS 中 BiCodec 的 optional 薄集成。它不复制
Spark-TTS 源码，也不在 import `anytrain.codec` 时导入 `sparktts`；只有调用
`BiCodec.from_pretrained()` 时才需要上游源码和依赖。

## 安装

`rookiejune/Spark-TTS` fork 提供标准 Python package metadata。直接安装固定 commit；该命令会同时
安装 `sparktts` 和 BiCodec runtime 依赖：

    python -m pip install "spark-tts @ git+https://github.com/rookiejune/Spark-TTS.git@cf5302b0da797aeffc18af172de4dda0bb6ab718"

安装不需要手动 clone 或设置 `PYTHONPATH`。fork 的 package metadata 不包含 Gradio Web UI；
BiCodec runtime 不依赖该 UI。

## 使用

    from anytrain.codec.bicodec import BiCodec

    codec = BiCodec.from_pretrained(
        cache_dir="/path/to/model-cache/bicodec",
        device="cuda",
    )

    codes = codec.tokenize(audio, sample_rate=48000)
    audio_out = codec.detokenize(codes)

    semantic = codes.semantic
    acoustic = codes.acoustic

`audio` 使用 `[batch, 1, time]` 单声道 waveform。wrapper 会重采样到 Spark-TTS 的
16 kHz，并用上游 Wav2Vec2 特征抽取器生成 BiCodec 所需的 `feat`。默认把同一段音频裁成
reference clip 来提取说话人全局 token；如果目标音频和参考音频不同，可以显式传入：

    codes = codec.tokenize(
        target_audio,
        sample_rate=target_sample_rate,
        ref_audio=prompt_audio,
        ref_sample_rate=prompt_sample_rate,
    )

## Token 契约

Spark-TTS BiCodec 的主边界不是 `FrameCodec` 的单个 codes Tensor，而是：

- `semantic`：随时间变化的语义 token，由 BiCodec quantizer 产生。
- `acoustic`：reference audio 产生的固定长度全局声学 token。

规范 `SemanticAcousticCodec` 接口中，`tokenize()` 返回 `SemanticAcousticCodes`：

- `semantic` 使用 `[batch, time, 1]`。
- `acoustic` 使用 `[batch, slot, 1]`，released Spark-TTS-0.5B 固定为 32 个 slot。
- `acoustic_layout` 是 `FIXED_LENGTH`；32 表示整段 reference 的固定 slot 数，不是
  `32 token/s`，也不是 32 个 codebook。

上游原生 global tensor 是 `[batch, quantizer, slot]`。wrapper 在 `tokenize()` 边界转换为
`[batch, slot, codebook]`，在 `detokenize()` 时转回上游布局。released config 使用一个 FSQ
quantizer，因此 `acoustic_codebook_sizes` 只有一项。reference 配置当前裁成 6 秒，再压缩成固定
32 slots；slot 数不随 reference 秒数变化。

BiCodec 提供 `tokenize()` / `detokenize(codes)` structured 接口，以及同一
`SemanticAcousticCodec` 契约中的 acoustic feature boundary：`acoustic_codes_to_features()` 返回 `[batch, 32, latent]`
的 global slot 特征，`decode_features()` 接收同样的独立 acoustic 轴。不要把 global slot
广播到 semantic frame，也不要把 slot 解释成 codebook 轴。FM 因此可以在 32 个 slot 上做
flow matching；semantic condition 由训练侧 pooling 后广播到这 32 个输出位置。

## Checkpoint 与缓存

默认模型仓库是 `SparkAudio/Spark-TTS-0.5B`。`ensure_bicodec_assets()` 只下载 BiCodec、
`wav2vec2-large-xlsr-53` 和根 `config.yaml`，避免为 codec tokenization 拉取完整 LLM 权重。

缓存路径优先级：

1. 显式传入 `cache_dir=...`
2. `HF_HOME/bicodec`
3. 如果 `HF_HOME` 未设置，anytrain 会设置从 `ANYTRAIN_HOME` 派生的 Hugging Face 缓存，
   再使用 `HF_HOME/bicodec`

已有本地模型目录时可以跳过下载：

    codec = BiCodec.from_pretrained(
        model_dir="/path/to/Spark-TTS-0.5B",
        device="cuda",
        local_files_only=True,
    )
