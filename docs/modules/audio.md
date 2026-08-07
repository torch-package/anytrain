# `anytrain.audio` Design

## 定位

`anytrain.audio` 收拢 waveform 离散化相关的 optional 组件。它把方向 capability、双向 codec
组合和 frame-sequence BPE 放在同一功能域，同时保持各自的依赖 extra 与 lazy import。

```text
waveform
  → AudioTokenizer          # anytrain.audio.tokenizer
  → codes
  → (optional) CodecBPE     # anytrain.audio.bpe
  → model ids / idspace
codes
  → AudioDetokenizer        # anytrain.audio.detokenizer
  → waveform

AudioCodec = AudioTokenizer ∩ AudioDetokenizer   # anytrain.audio.codec
```

`AudioCodec` 是 tokenizer 与 detokenizer 的组合视图，通常共享同一 backend 实例。BPE 只消费
离散 codes，不加载神经网络 codec。

## 子模块

- [`anytrain.audio.tokenizer`](../codec.md)：`AudioTokenizer`、`load_audio_tokenizer`。
- [`anytrain.audio.detokenizer`](../codec.md)：`AudioCodeSpec`、`AudioDetokenizer`、
  `load_audio_detokenizer`。
- [`anytrain.audio.codec`](../codec.md)：protocol、registry、preset、concrete backend。
- [`anytrain.audio.bpe`](bpe.md)：`CodecBPE` / `EvalStats`。

根包 `import anytrain.audio` 保持轻量，不导出跨子模块快捷别名。

## 依赖

- waveform 工具：`audio` extra（torchcodec / torchaudio），不安装 codec 模型。
- concrete codec backend：`dac` / `longcat` / `bicodec` / `unicodec` 等。
- BPE：`bpe` extra。
