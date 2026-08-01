# `anytrain.module` Design

## 定位

`anytrain.module` 放 task-agnostic 的 `torch.nn.Module` 积木，用于下游普通 LightningModule 显式组合。

它不是模型 zoo，也不提供完整任务网络。组件必须保持小而清楚，只依赖 core torch、einops 或对应 optional extra。

## 当前状态

`anytrain.module` 顶层只导出少量稳定组件：

- `ADTConfig`
- `ADTStats`
- `AdaptiveDirichletTempering`
- `adt_stats_metrics`
- `ADT` 兼容别名
- `DiT`
- `DiTAttentionBackend`
- `DiTConditionState`
- `DiTConditionType`

子包显式导出其它组件：

- `anytrain.module.moe.routing`：density-neutral `ExpertRouting`、校验和 mask helper；`DenseExpertWeights` 是 probability-tensor 辅助视图。
- `anytrain.module.moe.conv`：`ConvRouter1d` / `ConvRouter2d`（只产 logits）、partitioner / shape helpers、`MultiScalePool1d`。
- `anytrain.module.moe`：统一入口 `MoE(num_experts=..., expert=Expert, weight=(E,...), bias=..., grid=...)`；Expert 对象负责 `__call__` / `bias_kind` / `bind(grid,...)`；`Grid1d` 表达 1D segment intent，`NativeGrid` 表示不切分的 native-resolution 路径；`expand_param` 做 E 轴扩维；`expand_expert` 仅便利 stock `Linear/Conv*`，不是通用 Module→Expert 推断。另有 `CallableExpert`、`linear` / `Conv*Expert`、`LinearRouter`、`NativeGrid`、`MoEOutput`、`ExpertRouting`、MoE aux-loss / metrics helpers。
- `anytrain.module.quantization`：`DEFAULT_FSQ_LEVELS`、`default_fsq_levels`、`FSQConfig`、`FiniteScalarQuantizer`、`VQConfig`、`EmbeddingVectorQuantizer`、`GVQConfig`、`GroupedVectorQuantizer`、`RVQConfig`、`ResidualVectorQuantizer`、`AGRVQConfig`、`AutoGroupResidualVectorQuantizer`、`QuantizerProtocol`、`ResidualQuantizerProtocol`、`QuantizerType`、`QuantizeOutput`、`QuantizationLoss`、`QuantizationUsage`、`quantization_usage`、`quantization_usage_metrics`、`codebook_usage`、`valid_indices_by_codebook` 和 `quantization_loss_metrics`。
- `anytrain.module.qwen`：Qwen3 builder、resolver、experimental `QwenMTPCodebookPredictor` 和 `top_p_filter`。
- `anytrain.module.idspace`：`Layout` 和 `Embedding`。

`anytrain.module.moe.conv` 还公开 `eca_kernel_size`，用于按 channel 数计算 router 的 odd
kernel size。

ADT 用于 MoE/router logits 的自适应温度缩放和专家利用率统计，适合跨 audio、vision、text 等项目复用。
公开构造面是 `AdaptiveDirichletTempering(num_experts, *, warmup_steps=0,
min_temperature=None, max_temperature=None)`。dispersion / EMA / Minka / Gumbel /
temperature_groups / sync 等实现旋钮先留在类私有默认值，需要时再提升为公开参数。
统计默认像 BatchNorm 一样只用本地 batch；跨卡同步以后若提供，会走 SyncBatchNorm 风格的显式 API。
统计收集路径会先把 FP16/BF16 logits 升到 FP32，再计算用于统计的 softmax、计数和各阶矩，
避免长序列 reduction 溢出。forward 输出仍以调用方传入的 logits dtype 计算温度 softmax。

### ADT contract

`AdaptiveDirichletTempering.forward()` 的输入是 `(..., num_experts)` router logits，输出是同
shape、同 dtype 的 expert weight tensor，最后一维是和为 1 的概率分布。`mask` 只影响统计收集，
必须能 broadcast 到 `logits.shape[:-1]`，不会改变返回权重的 shape。

ADT 的职责边界是 logits calibration 和 router utilization statistics：它不选择 expert、不计算
load-balancing loss、不假设 batch schema，也不直接做 Lightning logging。下游如果需要负载均衡、
专家 dropout、top-k routing 或任务相关日志，应在自己的 `LightningModule`、loss 或 router 组合层里显式实现。

公开统计优先使用命名清楚的属性：`expert_utilization` 表示每个 expert 的运行期平均 router 概率，
`effective_experts` 表示 utilization 分布的 entropy perplexity，`temperature` 表示当前应用到 logits 的
per-expert 温度。`stats()` 返回纯 `ADTStats` 对象；`adt_stats_metrics()` 只把 ADT 独有的
variance、alpha 和 temperature 汇总成稳定数量的 scalar metric key，例如 `variance_mean`、
`alpha_std`、`temperature_min` 和 `temperature_max`。gate utilization / effective experts 必须由
MoE/gate routing 日志输出，ADT stats metrics 不提供 usage logging 入口。
`diagnostics()` 直接返回默认 ADT-only scalar summary key，和 `ADTStatsLoggerCallback` 配套；需要结构化张量时使用
`stats()`，或者读取 ADT module 的只读 `monitor` property 获取当前 `ADTStats` snapshot。
如果需要看完整 ADT per-expert 分布，使用 `ADTStatsLoggerCallback(log_histograms=True)` 记录 variance、
alpha 和 temperature 的 histogram，而不是为每个 expert 生成一条 scalar 曲线。
默认使用 `ADTStatsLoggerCallback()` 即可：callback 会从 `pl_module.named_modules()` 自动发现所有
`AdaptiveDirichletTempering` module，读取它们的 `monitor`，并用通用 monitor naming helper 从 module path
推断最短唯一日志名：单个 `encoder.router.gate` 记为 `adt/gate/temperature_mean`，如果同时存在
`encoder.router.gate` 和 `decoder.router.gate`，则分别记为
`adt/encoder.router.gate/temperature_mean` 和 `adt/decoder.router.gate/temperature_mean`；
需要覆盖时再传显式 mapping key 或 `ADTStatsMonitor(name=...)`。
`ADT` 短名保留为兼容别名，新代码和文档应优先使用 `AdaptiveDirichletTempering` 全名。

`anytrain.module.moe` 的公共模型是统一 `MoE`：`num_experts` 为顶层真相，核心契约是 `Expert` 对象 + 已是 `(E,...)` 的 `weight`/`bias` + 可选薄 `grid`。`expand_expert` 只对 stock `nn.Linear`/`nn.Conv*` 做几何拷贝和参数扩维，**不能**也不应用于自定义 `nn.Module` 推断。Conv 与 ConvTranspose 共享 `expand_param`、`Grid1d` 和「restore 后再加 bias」；weight 布局与 restore（concat vs OLA）由各自 Expert.`bind` 选择。设计见 [`moe.md`](moe.md) 与 [`moe-routing.md`](moe-routing.md)。

- Token MoE：`MoE(..., expert=linear, weight=(N, out, in), top_k=2)`；需要对齐某个 `nn.Linear` 时可用 `expand_expert` 便利构造。
- Dynamic conv：`MoE(..., expert=Conv1dExpert(...), weight=..., grid=Grid1d(...), top_k=None)`；对齐某个 `nn.Conv1d` 时可用 `expand_expert`，自定义 cell 算子用 `CallableExpert`（native-resolution）或自实现 `Expert.bind`。
- `moe_loss()` 和 routing metrics 是显式 helper；MoE forward 不自动改训练目标或打日志。

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
- 缺少 `transformers>=4.51.0` 或版本不含 Qwen3 时，builder 会抛出明确的 `ImportError`。

`anytrain.module.qwen.mtp` 提供 experimental `QwenMTPCodebookPredictor`，用于复用 Qwen3 组件组合
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
  stage 使用 `num_groups` 个同尺寸 learned group codebook，默认 `num_groups=2`；
  配置中的 `group_size` 是单个 group 的大小；对外 flat `codebook_size` 是
  `group_size ** num_groups`，输出 `codebook_dim` 是配置值乘以 `num_groups`。
  `input_dim` 必须能被 `num_groups` 整除，推理可用 `num_active_codebooks` 截断 stage 数，
  训练 dropout 用 `indices == -1` 和 `active_codebook_mask == False` 标记未启用 stage。
- 量化输出统一为 `QuantizeOutput`，离散整数用 `indices`，连续 codebook 空间向量用 `codebook_vectors`。
- `quantization_usage()` / `codebook_usage()` 只基于 `indices` 和 `active_codebook_mask`
  统计 codebook 使用率、perplexity、dead code ratio 和 active ratio；`quantization_usage_metrics()`
  和 `quantization_loss_metrics()` 负责生成稳定相对 metric key。它们不依赖 Lightning，也不会触发
  quantizer forward；Lightning callback 只负责 monitor 解析、分布式 reduce 和日志 transport。
- 单 codebook quantizer 实现 `QuantizerProtocol`；支持 `num_active_codebooks` 的 residual quantizer 额外实现 `ResidualQuantizerProtocol`。
- 所有 quantizer 暴露 `codebook_sizes`，单 codebook 为 `(codebook_size,)`，residual
  / multi-stage quantizer 为每个 stage 的 flat id 空间大小。
- RVQ dropout 下 inactive codebook 使用 `indices == -1` 和 `active_codebook_mask == False` 标记，不能只靠 `codebook_vectors` 判断有效性。

迁移设计见 [`docs/quantization-migration.md`](../quantization-migration.md)。第一版不迁入 audio codec、model zoo 或 deepaudio 的 projector/MoE 体系。

多模态 token id space 和 block embedding 路由放在 `anytrain.module.idspace`，
设计见 [`anytrain.module.idspace`](idspace.md)。

## 依赖策略

`torch` 和 `einops` 是默认依赖。`einops.rearrange` 用于让动态分组卷积的 batch/channel/kernel shape 变换保持可读。

`anytrain.module` 后续如有额外三方依赖，使用 `module` extra 管理：

```bash
python -m pip install "anytrain[module]"
```

当前 Qwen3 复用层和 `QwenMTPCodebookPredictor` 需要 `module` extra 中的
`transformers>=4.51.0`。ADT、MoE conv 和 quantization 不需要 optional 依赖；
`import anytrain` 和 `import anytrain.module` 不会主动导入 `transformers`。
