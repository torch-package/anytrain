# Codec Environments

`anytrain.audio.codec` 的边界是 optional thin wrapper：core 包不安装具体 codec
上游依赖，wrapper 只在用户显式调用时导入上游包。codec 依赖如果能和主训练环境共存，
可以放进 extra；如果上游 pin 旧版 torch、Python 或 pip，就使用独立环境先离线生成
codes/cache，再由主训练环境读取产物。

## GLM-4 online tokenizer

`anytrain.audio.codec.glm4.GLM4Tokenizer` 采用目前经过 exact token-ID 审计的官方 API 和数值契约：

- source：通过 `GLM4_VOICE_SOURCE_ROOT` 显式传入 GLM-4-Voice checkout 或用户 fork。loader 不读取
  Git remote/HEAD，也不限制 commit；它校验 `speech_tokenizer` 必需文件、导入来源和
  `WhisperVQEncoder.from_pretrained()` API，随后用实际模型和 feature extractor 校验数值契约；
- weights：`zai-org/glm-4-voice-tokenizer` commit
  `a5f2404e63c84e92f5238908e1706316324ebafa`；
- runtime：严格 `transformers==4.44.1`，官方
  `speech_tokenizer.modeling_whisper.WhisperVQEncoder` 加
  `transformers.WhisperFeatureExtractor`；
- contract：16 kHz mono waveform，CPU float32 fbank、禁用 TF32、STFT 200-sample guard、
  1280-sample padding multiple，输出 `torch.long [batch, frame, 1]`，ID 范围 `[0, 16384)`；
- batch：wrapper 对一个采样率分组执行一次 padded feature extraction 和一次 model
  forward；每条 waveform 在重采样前按真实长度裁剪，有效 token 长度由 feature
  `attention_mask` 依次按 convolution stride 和 pooling stride 下采样得到。固定 16 kHz
  输入下等价于 `ceil(samples / 1280)`；尾部 padded IDs 不进入返回的逻辑序列。短音频
  仍使用一次 padded batch forward；超过官方 Whisper 1500 source-position 上限的长音频
  按连续的 478720-sample（374-token）安全块分轮编码，再按原 waveform 顺序拼接。该边界
  同时满足 1280-sample token 网格和 200-sample STFT guard，因此不会丢失或重复 token；
- identity：真实 batch 和长音频 chunking 的有效位置 ID 允许因 batch shape 的浮点数值路径而
  不同于历史 single-sample extraction。registry identity 使用带
  `anytrain-batched-chunked-v1` 后缀的执行 revision；下游 provider/materialization identity
  必须包含该 identity，不得与旧 GLM-4 store 混合 resume。

默认 `anytrain` py312 环境当前使用 Transformers 4.57.6，且 BiCodec extra 要求
`transformers>=4.46`、module extra 要求 `transformers>=4.51.0`。因此不要为 GLM-4 增加一个会让
resolver 产生冲突的组合 extra，也不要把 fake 单测当作新版本兼容性证据。推荐部署是独立
Transformers 4.44.1 GLM producer 持续生成/publish raw GLM codes，主训练环境同时消费这些 codes
并运行 BiCodec target。若需要训练进程内 fallback，必须先在候选联合环境用固定真实音频完成
prepared/online exact ID parity、batch-single parity 和吞吐/显存 gate。

## Fudan smoke 2026-07-11

测试目录：

- `122:/mnt/pami202/zhuyin/dynamic/debug/anytrain-codec-smoke`
- `121:/mnt/pami202/zhuyin/dynamic/debug/anytrain-codec-smoke`

结果：

- DAC：`122` 的 `py312` 环境已有 `dac==1.0.0`，`torch==2.7.0+cu128`。
  随机初始化的小 DAC 模型在 CPU 上完成 encode/decode，说明 DAC 代码路径可与当前
  py312/torch2.7+ 环境共存。官方预训练权重在 GitHub release，`121` 上直接下载
  `descriptinc/descript-audio-codec` release 30s 超时；真实权重 smoke 需要预置缓存
  或可用下载镜像。
- UniCodec：本地 `UniCodec` 源码在 `py312` 下能被加入 `PYTHONPATH`，但导入
  `Unicodec` 时缺 `fairseq`。`fairseq` 在 `py312` + pip 26 下解析/构建失败；
  在 isolated `py39` 中降到 pip 24.0 后，metadata 问题消失，但 resolver 会继续拉取
  大体积 torch wheel。UniCodec 不应进入 core；如果需要真实 smoke，先在 codec 专用
  环境里固定 torch，再安装 `fairseq` 等旧依赖。
- Stable Codec：`stable-codec==0.1.2` 在 `py312` dry-run 中要求
  `torch==2.4`、`torchaudio==2.4`、`stable-audio-tools==0.0.17`，并在
  `pandas==2.0.2` 处开始源码构建后失败。`121` 上 isolated `py39` 环境已完成真实安装：
  `python==3.9.25`、`stable-codec==0.1.2`、`torch==2.4.0+cu121`、
  `torchaudio==2.4.0+cu121`、`torchvision==0.19.0+cu121`，CUDA 可见
  `NVIDIA A100-PCIE-40GB`。安装前需要先处理 `pypesq==1.2.4` 的旧构建链：
  `numpy==1.23.5`、`Cython`、`pip==24.0`、`setuptools==59.8.0`，再用
  `CFLAGS="-fcommon" pip install --no-build-isolation pypesq==1.2.4`。Stable Codec
  应使用独立 py39/torch2.4 环境，不放进 `anytrain` extra 自动安装。安装匹配
  torch 2.4/CUDA 12.1 的 `flash-attn==2.7.4.post1` 后，已在 `121` A100 上用统一接口
  完成未启用 posthoc 的真实 1 秒静音 roundtrip：native
  `codebook_sizes == (24137569,)`，codes shape 是 `(1, 25, 1)`，decode 输出
  `(1, 1, 16000)` 且数值有限。

## Policy

- `anytrain` core 继续保持宽 Python 约束和当前 torch 主线，不为旧 codec 整体降级。
- DAC 已作为普通 optional codec wrapper 接入；wrapper 使用显式 `cache_dir` 或
  `ANYTRAIN_HOME/dac`，不调用上游硬编码到 `Path.home()/.cache/descript/dac` 的下载入口。
- 远程 checkpoint/snapshot 下载默认要求可复现 pin：Hugging Face 路径传
  `revision=<commit-sha>`，DAC 路径需要 SHA-256 元数据；旧式浮动/未校验下载必须显式 opt in。
- UniCodec wrapper 保持 optional；不要让 pip 在主环境里自由解析 `fairseq` 和 torch。
- Stable Codec wrapper 保持 optional；文档要求用户在独立兼容环境中安装上游包和
  FlashAttention。
- GLM-4 wrapper 保持 tokenizer-only；允许 source fork，但必须满足已验证 API/数值契约；HF 权重
  revision 和 Transformers 版本继续固定。与主环境依赖冲突时使用独立 producer，不为方便安装
  静默放宽版本约束。

## UniCodec smoke 2026-07-13

`121` 已完成真实 speech encode/decode。输入沿用 LongCat 和 Stable Codec 对比使用的
6 秒单声道音频：16 kHz 文件 SHA-256 为
`4961d000272d1963372b7f98e79452368bca9a7b54dc0c0b08717004f12d4db0`，统一重采样并
重新读取为 24 kHz、144000 samples 后传给 UniCodec。

运行环境和结果：

- Python 3.9，`torch==2.4.0+cu121`，`torchaudio==2.4.0+cu121`，
  `fairseq==0.12.2`，GPU 为 A100 40 GB。
- checkpoint 为 `Yidiii/UniCodec_ckpt/unicode.ckpt`，SHA-256 为
  `36c387bf6385b46420c1428eefbc3dedef24dfc5e7197b9d08e2e26260820096`。
- speech domain `0`、bandwidth id `0` 生成 codes `[1, 450, 1]`，码本大小 16384。
- codes 反量化后的 features 与 encode 返回 features 的最大绝对误差为
  `5.96e-08`，说明离散 codes roundtrip 与模型量化输出一致。
- decode 输出为 24 kHz、144000 samples；对齐输入后的 MSE 为 `0.0009833`，SNR 为
  `4.66 dB`。音频与完整 metadata 放在顶层
  `debug/codec-reconstruct-compare-20260712/`，不进入项目仓库。

真实 smoke 同时发现并修复了 `rookiejune/UniCodec` fork 的 SimVQ 反量化边界：
`Unicodec.codes_to_features()` 现在通过 quantizer 的公开 `decode()`，`SimVQ1D.decode()`
使用与 encode 相同的 projected codebook。

## Codec speech evaluator 2026-07-13

`121` 上从 `wmt19_tts` train split 选择 target index 310：
`But progress is not to be confused with victory.`。原始音频是 24 kHz、2.56 秒，
SHA-256 为 `51411b4ed815e23481d2ffffff972d02dbcab1fec57bf98ae9f4fdb77f769f15`。
筛选时要求原始音频在 `large-v3-turbo`、English、temperature 0 配置下达到
WER 0、BLEU 100、chrF 100；三个 codec 都从保存后的同一个 PCM16 WAV 读取输入。

codec 配置：

- LongCat：`16k_4codebooks`，输出 16 kHz。
- Stable Codec：`speech-16k`，`normalize=False`，输出 16 kHz。
- UniCodec：speech domain `0`、bandwidth id `0`，输出 24 kHz。

SpeechEvaluator 使用同一个 `large-v3-turbo` ASR 实例和
`tarepan/SpeechMOS:v1.2.0` 的 `utmos22_strong`：

| Audio | WER | BLEU | chrF | UTMOS |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.0000 | 100.0000 | 100.0000 | 4.4607 |
| LongCat | 0.0000 | 100.0000 | 100.0000 | 4.4808 |
| Stable Codec | 0.1111 | 86.3340 | 83.0299 | 3.8711 |
| UniCodec | 0.0000 | 100.0000 | 100.0000 | 4.3878 |

Stable Codec 的转写在最后一个词变成 `Vig-`；原始音频、LongCat 和 UniCodec 的转写与
reference 一致。完整 WAV、codec metadata、候选筛选结果和 evaluator JSON 位于顶层
`debug/codec-speech-eval-20260713/`。这里只验证单条样本，不能据此推导总体 codec 排名。

## Stable Codec py39 recipe

`121` 已验证环境位置：

```bash
base=/mnt/pami202/zhuyin/dynamic/debug/stable-codec-py39
python=$base/env/bin/python
```

推荐复现步骤：

```bash
export TMPDIR=$base/tmp
export PIP_CACHE_DIR=$base/pip-cache
export CONDA_PKGS_DIRS=$base/conda-pkgs

conda create -y -p "$base/env" python=3.9
"$python" -m pip install pip==24.0 setuptools==59.8.0 wheel Cython numpy==1.23.5
CFLAGS="-fcommon" "$python" -m pip install --no-build-isolation pypesq==1.2.4
CFLAGS="-fcommon" "$python" -m pip install stable-codec==0.1.2
```

验证命令：

```bash
CUDA_VISIBLE_DEVICES=0 "$python" - <<'PY'
import inspect
import torch
from stable_codec import StableCodec

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(inspect.signature(StableCodec))
print(inspect.signature(StableCodec.encode))
print(inspect.signature(StableCodec.decode))
print(inspect.signature(StableCodec.set_posthoc_bottleneck))
PY
```

`stable-codec==0.1.2` 的上游 API：

- `StableCodec(model_config_path=None, ckpt_path=None, pretrained_model=None, device=torch.device("cpu"))`
- `encode(audio, posthoc_bottleneck=False, normalize=True, **kwargs)` 返回
  `(continuous_latents, tokens)`；native tokens 是 `(B, S, 1)` Tensor，posthoc tokens 是
  每个 stage 一个 `(B, S, 1)` Tensor 的 list。
- `decode(tokens, posthoc_bottleneck=False, **kwargs)` 在 native 模式接收 Tensor，在 posthoc
  模式接收该 list，返回 `(B, C, L)` waveform。
- `set_posthoc_bottleneck(stages)` 支持 preset 字符串或 stage 配置。

因此 `anytrain.audio.codec.stable_codec.StableCodec` 继续作为 optional thin wrapper：
`from_pretrained()` / `from_config()` 只负责加载上游模型，`encode()` 返回统一的
`[batch, frame, codebook]` codes，`encode_latents()` 暴露上游 latent/code 边界，
`decode()` 接收 codes。wrapper 在 posthoc 模式负责上游 list 与公共 Tensor 间的双向转换，
native `codebook_sizes` 从 `model.model.bottleneck.quantizer` 读取；wrapper 默认启用
`1x46656_400bps` posthoc preset，得到 `(46656,)`。显式传入 `posthoc_bottleneck=None`
时，speech 模型使用 `(17^6,) == (24137569,)` 的 native FSQ。现有 wrapper
签名和上游 `0.1.2` 对齐，不需要为了 Stable Codec 修改 `anytrain` 主环境或
`pyproject.toml`。
