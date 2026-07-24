# `anytrain.module` Design

## 定位

`anytrain.module` 放 task-agnostic 的 `torch.nn.Module` 积木，用于下游普通 LightningModule 显式组合。

它不是模型 zoo，也不提供完整任务网络。组件必须保持小而清楚，只依赖 core torch、einops 或对应 optional extra。

## 当前状态

`anytrain.module` 顶层只导出少量稳定组件：

- `ADTConfig`
- `AdaptiveDirichletTempering`
- `ADT` 兼容别名
- `DiT`
- `DiTAttentionBackend`
- `DiTConditionState`
- `DiTConditionType`

子包显式导出其它组件：

- `anytrain.module.dynamic_conv`：`DynamicConv1d`、`DynamicConv2d`、`DynamicConvTranspose1d`、`ADTRouter1d`、`ADTRouter2d`、`MultiScalePool1d`。
- `anytrain.module.quantization`：`DEFAULT_FSQ_LEVELS`、`default_fsq_levels`、`FSQConfig`、`FiniteScalarQuantizer`、`VQConfig`、`EmbeddingVectorQuantizer`、`GVQConfig`、`GroupedVectorQuantizer`、`RVQConfig`、`ResidualVectorQuantizer`、`AGRVQConfig`、`AutoGroupResidualVectorQuantizer`、`QuantizerProtocol`、`QuantizerType`、`QuantizeOutput` 和 `QuantizationLoss`。
- `anytrain.module.qwen`：Qwen3 builder、resolver、`QwenMTPCodebookPredictor` 和 `top_p_filter`。
- `anytrain.module.idspace`：`Layout` 和 `Embedding`。

`anytrain.module.dynamic_conv` 还公开 `eca_kernel_size`，用于按 channel 数计算 router 的 odd
kernel size。

ADT 用于 MoE/router logits 的自适应温度缩放和专家利用率统计，适合跨 audio、vision、text 等项目复用。
统计收集路径会先把 FP16/BF16 logits 升到 FP32，再计算用于统计的 softmax、计数和各阶矩，
避免长序列 reduction 溢出；启用 distributed stats 同步时也使用 FP32 collective。forward
输出仍以调用方传入的 logits dtype 计算温度 softmax。

Dynamic Conv 用于按样本或按分段动态组合 expert convolution kernel。它保持 task-agnostic：

- 核心卷积只接收外部 `router`，或通过 `forward_manually(x, expert_weights)` 显式传入权重。
- 默认 `ADTRouter1d` 只依赖 torch 原生层和 `AdaptiveDirichletTempering`。
- 当前提供 1D conv / transposed conv，以及用于 Conv2d subsampling 的最小
  `DynamicConv2d`；2D transpose、segment/chunk 和 causal padding 等能力等真实下游
  需求明确后再补。
- router 输出被视作 expert 权重，不在 `DynamicConv1d` 内猜测 logits/softmax 语义。

`anytrain.module.qwen.qwen3` 是 Hugging Face Qwen3 的薄复用层，不重新实现
transformer block，也不要求下游维护一份庞大的项目级 config：

- `anytrain.module.qwen` 导出 builder 和 resolver；`import anytrain.module` 不导入 Qwen 子模块。
- `anytrain.module.qwen.qwen3.Qwen3RMSNorm`、`Qwen3MLP`、`Qwen3Attention` 等名字通过
  module `__getattr__` 按需解析到
  `transformers.models.qwen3.modeling_qwen3`。
- `build_qwen3_mlp()`、`build_qwen3_attention()`、`build_qwen3_decoder_layer()` 等
  builder 只暴露对应模块初始化需要的少量参数，内部临时生成 Hugging Face
  `Qwen3Config`。
- 如果下游确实要构建完整 HF model，可以显式调用 `make_qwen3_config()` 或
  `build_qwen3_model()`；普通 codec 组合应优先依赖更窄的 builder。
- 缺少 `transformers` 或版本不含 Qwen3 时，builder 会抛出明确的 `ImportError`。

`anytrain.module.qwen.mtp` 提供 `QwenMTPCodebookPredictor`，用于复用 Qwen3 组件组合
“跨 frame temporal AR + frame 内 codebook MTP”的离散码本预测器：

- 输入是外部上游已经准备好的 `[batch, frame, condition_dim]` condition，不负责语义/声学
  codec、tokenizer 或完整任务网络。
- temporal Qwen 根据上一 frame 的首个 codebook 和当前 condition 预测当前 frame 的首个
  codebook；MTP Qwen 再在同一 frame 内依次预测剩余 codebook。
- `forward()` 返回每个 codebook 一个 teacher-forced logits tensor；`generate()` 提供同一
  预测顺序的自回归采样。
- `top_p_filter()` 从 `anytrain.module.qwen` 导出，供下游 generator 复用同一 nucleus sampling
  语义；它不进入 `anytrain.module` 顶层。

`anytrain.module.dit` 提供通用 `DiT` sequence backbone。它只负责把 noisy sequence、
flow time 和单一外部条件模式组合成输出 sequence，不接管 flow matching、batch schema 或
codec 逻辑：

- `condition_type=FRAME_FILM` 处理和输入 sequence 等长的 `condition: [B, T, C]`，通过每层
  FiLM/AdaLN 注入。
- `condition_type=FILM` 处理 batch-level `condition: [B, C]`，投影后 broadcast 到 frame 轴。
- `condition_type=CROSS_ATTN` 处理不等长 `condition: [B, S, C]` 和 `condition_mask: [B, S]`，
  通过 cross-attention 注入。
- `prepare_condition()` 将原始 condition 变成 `DiTConditionState`；cross-attention 会缓存
  每层 condition K/V，供 flow matching ODE sampling 的多个 step 复用。
- attention backend 支持 `EAGER` 和 PyTorch `SDPA`；`AUTO` 默认走 SDPA，由 PyTorch 在合适
  CUDA/dtype 场景下选择 flash / memory-efficient kernel。
- `forward_with_features()` 可返回指定中间层投影，供下游 REPA、蒸馏或诊断目标使用；具体
  loss 仍由下游或 `anytrain.framework.flow_matching` 组合。

Quantization 提供 task-agnostic 的 FSQ、embedding-table VQ、GVQ、RVQ 和 AGRVQ：

- `FiniteScalarQuantizer` 使用 `levels` 描述每个 scalar code dimension 的 level 数；它和普通 VQ 一样是单 codebook 接口，`indices` shape 为 `(...)`，`codebook_vectors` shape 为 `(..., codebook_dim)`。
- FSQ 默认使用 odd-only `levels` preset，推荐 odd `levels` 以保持 scalar grid 关于 0 对称；even `levels` 会触发 warning，但仍使用保留 0 码点的 zero-friendly grid。
- FSQ 的 `bound_scale` 控制进入 `tanh` 前的 latent 尺度；调大它可以缓解大幅值 projected latents 的边界饱和。
- `EmbeddingVectorQuantizer` 是 embedding table + nearest-neighbor VQ，实现名避免和 protocol 混淆。
- nearest-neighbor 查找会沿 latent 和 codebook 两个维度自动分块，确保任一临时比较矩阵不
  超过内部元素上限，限制长序列和超大码本组合时的峰值显存。
- `EmbeddingVectorQuantizer` 配置 `VQConfig(use_ema=True)` 时使用 `bincount` / `index_add_`
  以 FP32 聚合并保存统计，即使 quantizer 整体转换为 FP16/BF16 也不降低 EMA 精度；
  `torch.distributed` 已初始化时会在更新 EMA 和 codebook 前全局求和 counts/sums。初始 EMA
  state/codebook 必须已在 rank 间同步，且所有 rank 必须使用相同的 train/eval mode 和 active
  stage 数。RVQ 同时启用 EMA 和 per-vector dropout 时，本地没有 assignment 的 stage 仍以零
  counts/sums 参与 collective；如果该 stage 在所有 rank 都没有 assignment，则保持 EMA
  state/codebook 不变。如果可能被整 stage dropout 的 quantizer 含可训练参数，PyTorch DDP
  还需要设置 `find_unused_parameters=True`，因为该 rank 的参数不会进入本次 autograd graph。
- `GroupedVectorQuantizer` 是 learned product codebook，例如 `group_sizes=(90, 90)` 对外表现为 `codebook_size=8100` 的单 codebook，但内部只搜索两个 90-size group。
- `ResidualVectorQuantizer` 组合多个 embedding VQ，第一版要求统一 `input_dim`、`codebook_dim` 和 `codebook_size`；`latents_to_codebook_vectors()` 不触发 dropout、loss 或 EMA 更新。
- `AutoGroupResidualVectorQuantizer` 从 `anytrain.module.quantization` 显式导入。每个 residual
  stage 使用两个同尺寸 learned group codebook；配置中的 `codebook_size` 是单个 group 的大小，
  对外 flat `codebook_size` 是它的平方，输出 `codebook_dim` 是配置值的两倍。`input_dim` 必须为
  偶数，推理可用 `num_active_codebooks` 截断 stage 数，训练 dropout 用 `indices == -1` 和
  `active_codebook_mask == False` 标记未启用 stage。
- 量化输出统一为 `QuantizeOutput`，离散整数用 `indices`，连续 codebook 空间向量用 `codebook_vectors`。
- RVQ dropout 下 inactive codebook 使用 `indices == -1` 和 `active_codebook_mask == False` 标记，不能只靠 `codebook_vectors` 判断有效性。

迁移设计见 [`docs/quantization-migration.md`](../quantization-migration.md)。第一版不迁入 audio codec、model zoo 或 deepaudio 的 projector/MoE 体系。

多模态 token id space 和 block embedding 路由放在 `anytrain.module.idspace`，
设计见 [`anytrain.module.idspace`](idspace.md)。

## 依赖策略

`torch` 和 `einops` 是默认依赖。`einops.rearrange` 用于让动态分组卷积的 batch/channel/kernel shape 变换保持可读。

`anytrain.module` 后续如有额外三方依赖，使用 `module` extra 管理：

```bash
python -m pip install transformers
```

当前 Qwen3 复用层和 `QwenMTPCodebookPredictor` 需要 `module` extra 中的
`transformers`。ADT、Dynamic Conv 和 quantization 不需要 optional 依赖；
`import anytrain` 和 `import anytrain.module` 不会主动导入 `transformers`。
