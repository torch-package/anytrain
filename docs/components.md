# Components

## 设计原则

`anytrain` 的组件是给用户写 PyTorch/Lightning 训练代码时使用的积木，尤其服务普通 `LightningModule` 的实现。下游项目负责训练入口和配置装配；用户仍然控制训练逻辑，但可以选择 `anytrain` 提供的 module、loss、evaluator、optim、chat 和 framework。

组件应该是显式可组合的对象，而不是隐藏在自定义 `LightningModule` 基类里的行为。一个组件如果会改变训练流程、文件系统状态或日志行为，应通过明确的 helper、callback 或 `pl_module.__init__` 参数暴露。

组件按依赖拆分：

- core 组件默认可用。
- optional 组件在对应子模块中提供。
- optional 子模块缺依赖时应给出实际缺失的包和可执行安装命令，而不是影响 core import。

## Logging

LightningModule 侧 logging helper 是 core。

core 提供：

- `LightningLogMixin`：prefixed dict、audio 和 figure logging helper。
- `LossSummaryCallback`：用有界 device 窗口汇总 training outputs 中的 total loss 和命名
  `LossItem`，并在训练结束时归约跨 rank 窗口均值。
- `LossItemLoggerCallback`：按下游提供的 label adapter 汇总并记录命名 `LossItem`。
- `GradientProbeLoggerCallback`：比较 `loss@group` target 对指定参数 probe 的梯度 norm、log-ratio 和 cosine。
- `LossTimeBucketLoggerCallback`：按 `LossItem.details` 中的 time-like tensor 记录 histogram 和 bucketed loss。
- `ADTStatsLoggerCallback`：记录下游返回的 `AdaptiveDirichletTempering` / `ADTStats` monitor，默认只打 ADT 独有的 variance、alpha 和 temperature；直接返回 ADT module 时按 `pl_module.named_modules()` 自动推断日志名。
- `CodebookUsageLoggerCallback`：按下游显式声明的 codebook indices 记录 usage、perplexity、dead code 和 active codebook 比例。
- `QuantizationLoggerCallback`：按下游显式声明的 `QuantizeOutput` 记录量化 loss，并复用 codebook usage 统计。
- `MoELoggerCallback`：从 training outputs 自动发现 `MoEOutput` / `ExpertRouting` / `MoEMonitor`，记录 MoE usage、gate probability、actual route weight、assignment counts、dropped-token ratio 的稳定 scalar summary、可选 histogram 和 aux-loss components，并在匹配到 ADT gate 时自动追加 ADT-only `adt/...` stats。
- `DebugCallback` 等训练调试 callback。
- `PerformanceCallback`：记录参数量、step time、硬件峰值算力和 MFU。

optional backend：

- wandb。
- 其他第三方 logger。

第三方 backend 是 optional；logger backend 由下游或 Lightning 原生配置负责创建。

## Perf

`anytrain.lightning.perf` 是 core 组件，负责训练效率观测的通用定义和 task-agnostic 计算，不接管 batch schema。

当前提供：

- `count_parameters()`：统计参数量或 trainable 参数量。
- `profile_forward_flops()`：用 PyTorch profiler 在代表性输入上估算 forward FLOPs。
- `training_flops_from_forward()`：把 forward FLOPs 转换为训练 step FLOPs，backward multiplier 必须显式可见。
- `infer_peak_flops()`：根据 GPU 型号和 compute dtype 查内置硬件表；下游 job 可覆盖 `hardware_peak_flops`。
- `model_flops_utilization()`：按 `model_flops_per_step / step_time / hardware_peak_flops` 计算 MFU。
- `PerformanceCallback`：在 Lightning 训练中记录性能元数据、optimizer step time 和 MFU；
  支持固定 step FLOPs，以及由 `FlopsProvider` 提供的动态 train-batch FLOPs。

`anytrain` 不内置任务的数据量语义。samples、tokens、frames、valid mask count 和详细 loss 由下游 `training_step` 按任务 schema 显式记录；需要把这些 units 作为训练时钟、LR schedule 和停止条件时，使用 `anytrain.lightning.schedule`。

## Loss

core loss 负责统一训练 step 里的主 loss 约束：

```python
loss = loss_fn(prediction, target)
```

主返回必须是可用于 `backward()` 的 scalar Tensor。需要额外日志时，loss 可以返回 `(loss, details)`；`details` 是可选的 mapping，不参与反传。

当前接口：

- `LossABC`：抽象基类，子类实现 `compute_loss()`，统一校验 scalar 主 loss 和可选 details。
- `LossGroup`：用 mapping/`ModuleDict` 组合多个 loss，返回 `(total, details)`。
- `LossItem`：保存按样本/行聚合的 loss 与可选 details，供下游训练 step 组合和记录。
- `MaskedFrameMSELoss` / `MaskedCosineAlignmentLoss`：对 `[B, F, D]` 表示执行 frame mask 后的
  MSE 或 cosine alignment。
- `MaskedCodebookCrossEntropyLoss`：对多 codebook logits 执行 masked CE，并可选返回逐 codebook
  top-1。
- `LossBalancerABC`：把多个命名 scalar loss 合成为一个 scalar total，可选返回 details；默认实现是 `MeanLossBalancer`，core 也提供 `FixedWeightLossBalancer` 和 `UncertaintyLossBalancer`。

规划接口：

- plain-config loss 装配 helper：按配置创建 `LossABC` / `LossGroup` 和 balancer，
  但不新增 `task` 命名空间，也不解释 batch 或训练 step。

optional loss：

- spectral / temporal audio loss。
- text/speech 领域 loss。

## Evaluator

core evaluator 负责统一 metric 返回格式：

```python
metrics = evaluator(prediction, target)
```

当前接口：

- `EvaluatorABC`：继承 `torch.nn.Module` 的无状态 evaluator 抽象基类，子类实现 `evaluate()`。
- `EvaluatorGroup`：用 `nn.ModuleDict` 组合返回 metric mapping 的 evaluator module，并处理 key 校验。
- `EvaluatorABC.update/compute/reset` 是可选状态生命周期；基类默认明确抛出
  `NotImplementedError`，具体 evaluator 按指标定义实现正确状态。
- `EvaluatorGroup` 逐个代理状态生命周期，不跳过无状态子 evaluator。

`anytrain.evaluator.text.TextComparisonEvaluator` 保存规范化文本并按完整 corpus 计算 BLEU、WER
和 chrF；分布式初始化后会在 `compute()` 聚合所有 rank 的文本。core 不对 batch scalar 做通用
平均，也暂不提供通用 stateful MAE/MSE/accuracy。

optional evaluator：

- audio/codec/speech evaluator。
- text evaluator。
- torchmetrics-backed 通用 evaluator。

## Optim

optim 负责 optimizer / scheduler 构造 helper，不接管训练流程。

当前提供：

- AdamW 参数组 helper：按标准 AdamW 或 Muon-eligible policy 拆分 decay 和 no-decay。
- Muon 参数组 helper：默认只把 hidden 2D weight 放入 Muon，其余参数走 AdamW；head 等特殊模块由用户显式传入排除。
- LoRA-Muon：核心接收显式低秩因子对；`anytrain.optim.peft` extra 提供 vanilla PEFT Linear adapter 收集和 optimizer factory。
- `CompositeOptimizer`：把 Muon 和 AdamW 包成一个 optimizer，方便 Lightning 和 scheduler 使用。
- LLM helper：按 `pretrain` / `cpt` / `sft` stage 生成 AdamW 或 Muon+AdamW optimizer，并提供 `constant` / `warmup_cosine` / `wsd` 命名 scheduler。

optim 不提供魔法 LightningModule 基类。下游在自己的 `configure_optimizers()` 里显式调用 helper。

## Module

`module` 是 core 组件，提供 task-agnostic 的 `torch.nn.Module` 积木。顶层只导出少量稳定组件；
dynamic conv、MoE、quantization、Qwen 和 idspace 从各自子模块显式导入。

适合放入：

- router/gating helper。
- 动态层。
- 不绑定 batch schema 或具体任务 step 的研究模块。

不适合放入：

- 完整模型 zoo。
- 项目私有网络结构。
- 需要 audio/text/speech 数据语义才能理解的模块。

当前提供：

- Adaptive Dirichlet Tempering，用于 MoE/router logits 的自适应温度缩放。
- `anytrain.module.moe.routing`：MoE routing 家族共享的 dense/sparse expert routing 表示、校验和 mask broadcast helper；不执行 expert，也不持有 task schema。
- `anytrain.module.moe.conv`：1D Dynamic Conv / Dynamic Conv Transpose 和 2D Dynamic Conv，用于按样本或按分段组合 expert kernel。
- `anytrain.module.moe`：统一 MoE 的 router、dense/top-k dispatch、layer、显式 aux loss 和 metrics；expert 模块由下游显式传入。
- `anytrain.module.quantization`：embedding、finite scalar、grouped、residual 和 auto-group residual quantizer，以及不依赖 Lightning 的 usage/perplexity 统计 helper。
- `anytrain.module.qwen`：依赖 `module` extra 的 Qwen3 builder 和 experimental MTP codebook predictor。

`anytrain.module.idspace` 也是 core 组件，维护 local/global token id 映射和 block embedding
路由，不把 tokenizer 或任务 special-token 语义塞进 tokenizer 或任务层。

## Chat

chat 是 optional general 组件，依赖 `chat` extra。

建议边界：

- chat 只负责按环境变量解析 provider 配置，并把显式 prompt 和实例内消息上下文发给指定 provider。
- 下游项目负责 prompt 模板、任务 schema、provider cache 观测和结果记录。
- 缺少环境变量或 provider 后端不可用时直接抛出明确错误。

## Framework

framework 是 optional/experimental 组件。这里的 framework 指研究训练范式组件，不是默认训练 app 层。

适合放入：

- flow matching。
- masked autoencoder。
- GAN 训练辅助。

不适合放入 core，因为这些是训练范式，不是每个 LightningModule 都需要的基础能力。

不适合放入：

- 要求下游继承特定 `LightningModule` 基类的完整任务框架。
- 隐式接管 optimizer、scheduler、batch 解释或 logging 的大封装。
- 只能服务单个项目的私有训练模板。

## Import 规则

推荐规则：

- `import anytrain` 不导入 optional 领域依赖。
- `import anytrain.lightning` 可以假设 torch/lightning 已安装。
- `import anytrain.module` 可以假设 torch/einops 已安装；dynamic conv、quantization、Qwen 和 idspace 使用显式子模块入口。
- `import anytrain.loss.spectral` 和 `import anytrain.loss.temporal` 当前只依赖 core torch。
- `import anytrain.optim` 可以假设 torch 已安装。
- `import anytrain.chat` 不应要求 chat extra；真实 provider 请求路径可以要求 `chat` extra。
- `import anytrain.tokenizer` 不加载 `tokenizers`；构造或训练 `CodecBPE` 时要求 `tokenizer` extra。
- `import anytrain.codec` 只暴露轻量 protocol、静态 audio preset spec 和 lazy registry，不加载具体
  backend；默认 tokenizer / detokenizer capability view 按 resolved preset identity 和 device 共享底层
  backend，加载模型时要求对应 extra。只有 metadata、没有通用 online adapter 的 preset 必须明确
  报告 loader 不可用。
- `import anytrain.tts` 不加载具体 backend；加载模型时要求对应 extra。
- optional 缺依赖时抛出明确错误，包含实际缺失的包和可执行安装命令。
