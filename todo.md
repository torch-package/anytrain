# TODO

只记录尚未实现或需要真实环境确认的需求。历史迁移记录以文档和 git 历史为准。

## Lightning

- 在 speech-to-speech 的 8-rank 长跑中验证通用 stall audit：同一 run 关联
  anytrain/anydataset 记录，确认 S2TT cost planning 冷窗、callback、checkpoint 和
  collective arrival skew 能由 bounded JSONL 直接区分；任务字段继续由下游通过
  `batch_report` 提供，不进入 core。
- 设计真正只读的 module backward observation 契约。当前 `BackwardObservation` 只作为占位声明，
  `ObservationCallback` 会 warning 后跳过；需要观察 loss 对参数的梯度时先使用
  `GradientProbeLoggerCallback`。未来方案不能替换 module output、插入 autograd view，或让各 rank
  使用不同的 forward/backward graph。

## Loss

- 实现 plain-config 友好的 loss 组合/工厂：只负责按配置装配 `LossABC` / `LossGroup`
  和 balancer，不引入 `task` 命名空间，也不解释 batch 或训练 step。
- 增加更多 balancer 策略，例如 deviation。
- 只有在依赖和复用需求明确后，再加入 `loss.text` / `loss.speech`。

## Evaluator

- 有明确下游复用需求时，再增加 `evaluator.metrics`：torchmetrics-backed 通用分类/回归
  evaluator，放 optional 子模块而不是 core。
- 有具体 codec/speech acceptance 复用需求时，再增加 `evaluator.audio`；不要只为单个实验把
  UTMOS、ASR 或 audio quality wrapper 提前放进 core。

## Module

- Qwen3：根据真实训练需求暴露 SDPA / flash-attention knobs。
- Qwen3：只有在 `anycodec` 需要比 Hugging Face `Qwen3Model` 更窄的契约时，再增加 codec-specific token-forward wrapper。
- Qwen3：KV cache 语义需要独立设计，不混进训练 forward。
- Adaptive Dirichlet Tempering：支持非均匀 expert prior，用于 expert capacity 或有意偏置的 routing target。
- Adaptive Dirichlet Tempering：增加 top-k / hard routing 支持，包括 gumbel top-k 和 straight-through 变体。
- Adaptive Dirichlet Tempering：暴露可选辅助 loss，例如 KL-to-prior、entropy bonus 或 load-balance loss，但不自动接进 task loss。
- Adaptive Dirichlet Tempering：扩展 `diagnostics()`，加入 entropy、perplexity、min/max load、dead expert count 和 load imbalance ratio，并补 skewed routing 测试。

## Framework

- 后续新增 framework 子模块时，需要明确依赖 extra、最小测试和至少一个下游复用方。
- masked autoencoder 等训练范式只有在跨项目复用明确后再迁入；没有 owner 前不作为 active todo。
- framework 不替用户写完整 `pl_module`，只提供训练逻辑可复用组件。


## Descript Audio Codec

- 在预置官方 checkpoint 的环境里用真实短音频跑一次 encode/decode smoke，确认
  `DAC.from_pretrained(local_files_only=True)`、统一 codes 轴和多 `n_quantizers` 配置。
