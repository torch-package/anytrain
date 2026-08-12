# Spark-TTS BiCodec

`anytrain.audio.codec.bicodec` 是对 SparkAudio Spark-TTS 中 BiCodec 的 optional 薄集成。它不复制
Spark-TTS 源码，也不在 import `anytrain.audio.codec` 时导入 `sparktts`；只有调用
`BiCodec.from_pretrained()` 时才需要上游源码和依赖。

## 安装

`bicodec` extra 会安装固定 commit 的 `rookiejune/Spark-TTS` fork；该 fork 提供标准
Python package metadata，并同时安装 `sparktts` 和 BiCodec runtime 依赖：

    python -m pip install "anytrain[bicodec]"

安装不需要手动 clone 或设置 `PYTHONPATH`。fork 的 package metadata 不包含 Gradio Web UI；
BiCodec runtime 不依赖该 UI。

## 使用

    from anytrain.audio.codec.bicodec import BiCodec

    codec = BiCodec.from_pretrained(
        cache_dir="/path/to/model-cache/bicodec",
        device="cuda",
    )

    from anytrain.audio.detokenizer import load_audio_detokenizer
    from anytrain.audio.tokenizer import load_audio_tokenizer

    tokenizer = load_audio_tokenizer("bicodec", device="cuda")
    detokenizer = load_audio_detokenizer("bicodec", device="cuda")
    codes = tokenizer.tokenize(audio, sample_rate=48000)
    audio_out = detokenizer.detokenize(codes)

`audio` 使用 `[batch, 1, time]` 单声道 waveform。wrapper 会重采样到 Spark-TTS 的
16 kHz，并用上游 Wav2Vec2 特征抽取器生成 BiCodec 所需的 `feat`。默认把同一段音频裁成
reference clip 来提取说话人全局 token；如果目标音频和参考音频不同，可以显式传入：

    semantic, global_codes = codec.encode_codes(target_audio, target_sample_rate)

## Token 契约

Spark-TTS BiCodec 使用两个独立 named streams：

- `semantic`：随时间变化的语义 token，由 BiCodec quantizer 产生。
- `global_codes`：reference audio 产生的固定数量全局 speaker token。

tokenizer 返回：

- `semantic` 使用 `[batch, time]`。
- `global` 使用 `[batch, unit]`，released Spark-TTS-0.5B 的
  `global_unit_length == 32`。32 表示整段 reference 的全局 unit 数，不是 `32 token/s`，
  也不是 32 个 codebook。

released Spark-TTS-0.5B 的 semantic quantizer 使用 8192 个 token，因此 semantic ID
范围是 `[0, 8192)`。speaker global quantizer 使用一组 `levels=[4, 4, 4, 4, 4, 4]`
的 FSQ，因此 `global_codebook_sizes == (4096,)`，global ID 范围是 `[0, 4096)`。
两个字段属于独立 ID space，不能用 global 的 4096 上界校验 semantic token。

离线 batching 只需要 source audio header 时，使用
`audio_planning_estimator("bicodec")` 从 source frame count/rate 估算 semantic units。该静态
estimator 复用 pinned Wav2Vec2 unpadded convolution contract，不加载 BiCodec；其 identity 绑定
backend revision、公式版本和 convolution 参数。不要用 nominal 50 Hz 四舍五入替代它。

上游原生 global tensor 是 `[batch, quantizer, slot]`。backend adapter 在 tokenizer 边界转换为
单码本 `[batch, slot]`。released config 使用一个 FSQ
quantizer，因此 `global_codebook_sizes` 只有一项。reference 配置当前裁成 6 秒，再压缩成 32 个
global units；unit 数不随 reference 秒数变化。

BiCodec detokenizer 的 `codes_to_features("global", codes)` 返回
`[batch, 32, latent]` 的 global unit 特征，`decode_features()` 接收同样的独立 global 轴。
不要把 global unit 广播到 semantic frame，也不要把 unit 解释成 codebook 轴。FM 因此可以在
32 个 global units 上做 flow matching；global feature generator 需要为每个 unit 构造 condition，
例如让 learned unit query 通过 masked cross-attention 读取完整 semantic memory。BiCodec wrapper
本身不规定训练侧 condition 结构。

## Checkpoint 与缓存

默认模型仓库是 `SparkAudio/Spark-TTS-0.5B`。`ensure_bicodec_assets()` 只下载 BiCodec、
`wav2vec2-large-xlsr-53` 和根 `config.yaml`，避免为 codec tokenization 拉取完整 LLM 权重。
远程下载默认要求显式传入 Hugging Face `revision=<commit-sha>`；若需要沿用旧的浮动分支
下载行为，必须显式传入 `allow_unpinned_revision=True`。

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
