# UniCodec

`anytrain.audio.codec.unicodec` 是对 `rookiejune/UniCodec` fork 的 optional 集成层。
它不进入 core import，只在用户显式调用相关 API 时加载上游包。

## 安装

当前不提供 `unicodec` extra。UniCodec 的旧 `fairseq`/torch 构建链在 Python `>=3.12`
环境中仍会在解析或构建阶段失败，因此真实 UniCodec backend 暂无受支持的安装组合。不要通过较低 Python 版本或
`PYTHONPATH` bridge 绕过 anytrain 的包约束，也不要在主训练环境中尝试让 resolver 自由改写
共享 Torch 依赖。

历史 smoke 的版本和结果保留在 `docs/codec-envs.md`，只作为上游行为证据。待旧依赖链完成
Python 3.12 兼容后，再恢复 `unicodec` extra 和可执行安装说明。

## 缓存路径

checkpoint 自动下载到缓存目录。路径优先级：

1. 显式传入 `cache_dir=...`
2. `HF_HOME/unicodec`
3. 如果 `HF_HOME` 未设置，anytrain 会设置
   `HF_HOME=${ANYTRAIN_HOME:-~/.anytrain}/huggingface`，再使用
   `$HF_HOME/unicodec`

远程服务器上推荐：

```bash
export ANYTRAIN_HOME=/mnt/pami202/zhuyin/.anytrain
```

默认 checkpoint 来自 Hugging Face 仓库 `Yidiii/UniCodec_ckpt` 的
`unicode.ckpt`。默认 config 使用 UniCodec fork 内打包的
`unicodec_frame75_10s_nq1_code16384_dim512_acousitic.yaml`。
远程下载默认要求显式传入 Hugging Face `revision=<commit-sha>`；若需要沿用旧的浮动分支
下载行为，必须显式传入 `allow_unpinned_revision=True`。

## 使用

```python
from anytrain.audio.codec.unicodec import UniCodec

codec = UniCodec.from_pretrained(
    device="cuda",
    domain="0",
    bandwidth_id=0,
)

codes = codec.encode(
    audio,
    sample_rate=24000,
)
audio_out = codec.decode(codes)

features, codes = codec.encode_features(
    audio,
    sample_rate=24000,
)
audio_from_features = codec.decode_features(features)
```

`audio` 使用统一的 `[batch, 1, time]` 单声道 waveform；输入不为 24 kHz 时 wrapper 会
通过公共 codec 重采样 helper 重采样，该路径需要安装 `audio` extra 中的 `torchaudio`。
输入已经是 24 kHz 时不需要 `torchaudio`。`codes` 是 `[batch, frame, 1]`，
`codebook_sizes == (16384,)`。

`domain` 在构造 codec 时固定，直接沿用
UniCodec 上游约定：`"0"` 表示 speech，`"1"` 表示 music，`"2"` 表示 general
audio。它选择的是单一 codebook 内的 domain-adaptive partition，不是独立码本。

`bandwidth_id` 沿用上游参数名，但默认 config 里的 `bandwidths` 都是 `6.6`：

```yaml
bandwidths: [6.6, 6.6, 6.6, 6.6]
adanorm_num_embeddings: 4
```

因此第一版只把它视为构造时固定的 decoder/backbone 条件 id，默认使用 `0`。不要把它理解为
已经验证可用的可变码率接口；后续如果确认 checkpoint 中 `0/1/2/3` 有明确语义，再把
它提升成更清楚的枚举。

UniCodec 的主路径遵循统一 codec 接口，不拆分或命名 codebook 语义。

上游实现内部还有连续 quantized features 边界。需要直接操作这个边界时使用
`encode_features()` / `decode_features()`；连续 features 统一为 `[batch, frame, dim]`。
普通 codec roundtrip 应优先使用
`encode()` / `decode()`。

`local_files_only=True` 可以在离线环境中只使用已有缓存。
