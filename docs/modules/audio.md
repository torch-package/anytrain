# `anytrain.audio` Design

## 定位

`anytrain.audio` 收拢 waveform 离散化相关的 optional 组件。它把 tokenizer、detokenizer 两个方向
ABC、concrete backend registry 和 frame-sequence BPE 放在同一功能域，同时保持依赖 extra 与 lazy import。

```text
waveform
  → AudioTokenizer          # anytrain.audio.tokenizer
  → codes
  → (optional) CodecBPE     # anytrain.audio.bpe
  → model ids / idspace
codes
  → AudioDetokenizer        # anytrain.audio.detokenizer
  → waveform

```

默认 preset 的 tokenizer 与 detokenizer adapter 可以在 registry 内共享缓存的 backend 实例；
这是加载实现细节，调用方只依赖各自的方向契约。BPE 只消费离散 codes，不加载神经网络 codec。

## 子模块

- [`anytrain.audio.tokenizer`](../codec.md)：`AudioTokenizer`、`load_audio_tokenizer`。
- [`anytrain.audio.detokenizer`](../codec.md)：`AudioCodeSpec`、`AudioDetokenizer`、
  `load_audio_detokenizer`。
- [`anytrain.audio.codec`](../codec.md)：静态 code spec、directional registry、preset 与 concrete backend；
  新调用方不依赖通用 `AudioCodec`。
- [`anytrain.audio.bpe`](bpe.md)：`CodecBPE` / `EvalStats`。

根包 `import anytrain.audio` 保持轻量，不导出跨子模块快捷别名。

## 依赖

- waveform 工具：`audio` extra（torchcodec / torchaudio），不安装 codec 模型。
- concrete codec backend：`dac` / `longcat` / `bicodec` 等；UniCodec 和 Stable Codec 目前只保留 wrapper 边界，暂不提供受支持 extra。
- BPE：`bpe` extra。
