# Module Design Index

这个目录记录 `src/anytrain` 顶层模块的设计约定。模块文档描述当前实现、稳定接口、边界和后续演进方向；如果代码行为发生变化，应同步更新对应文档。

## 模块列表

- [`anytrain` package root](package-root.md)：轻量公开导出和 import 边界。
- [`anytrain.observation`](observation.md)：module concrete-type registry、observation 类型、显式 reduction 和 step/EMA 状态契约。
- [`anytrain.lightning`](lightning.md)：LightningModule logging mixin、调试 callback 和性能观测 callback。
- `anytrain.lightning.schedule`：step/unit LR schedule spec、unit 进度、unit LR runtime 和停止控制；归入 [`anytrain.lightning`](lightning.md)。
- [`anytrain.lightning.perf`](perf.md)：参数量、FLOPs、硬件峰值算力和 MFU。
- [`anytrain.loss`](loss.md)：通用 loss 组件和组合器。
- [`anytrain.evaluator`](evaluator.md)：训练期 evaluator 接口和组合器边界。
- [`anytrain.optim`](optim.md)：optimizer、scheduler 和 LLM/Muon helper。
- [`anytrain.module`](module.md)：task-agnostic `torch.nn.Module` 积木和子模块入口边界。
- [`anytrain.module.moe.routing`](moe-routing.md)：density-neutral `ExpertRouting`、校验和 mask 语义。
- [`anytrain.module.moe`](moe.md)：统一 `MoE` 入口（`num_experts` / `expert` / `grid`、dense/top-k runtime）与 metrics。
- `anytrain.module.moe.conv`：conv router、causal `Conv1d` 模板；归入 [`anytrain.module.moe`](moe.md)。
- [`anytrain.module.idspace`](idspace.md)：统一 token id space 和 block embedding 路由。
- [`anytrain.tokenizer`](tokenizer.md)：int BPE。
- [`anytrain.codec`](../codec.md)：optional audio codec wrapper，以及 frame、semantic-acoustic 与 semantic-global capability 契约。
- [`anytrain.tts`](../tts.md)：TTS 公共协议和 optional backend adapter。
- [`anytrain.chat`](chat.md)：环境变量驱动的可选大模型调用入口。
- [`anytrain.framework`](framework.md)：可选研究框架层。

## 关联文档

- [`docs/architecture.md`](../architecture.md)：总体分层、运行链路和包边界。
- [`docs/components.md`](../components.md)：训练组件的 core/optional 分层。
- [`docs/lightning.md`](../lightning.md)：Lightning 层的更完整设计。
- [`docs/flow-matching-design.md`](../flow-matching-design.md)：Facebook flow matching 的可组合再封装设计。
- [`docs/gan-framework-design.md`](../gan-framework-design.md)：参考 `deepaudio` 的 GAN framework 迁移设计。
- [`docs/quantization-migration.md`](../quantization-migration.md)：从 `deepaudio` 迁入量化组件的设计计划。
- [`docs/codec-envs.md`](../codec-envs.md)：codec optional 依赖环境和 Fudan smoke 记录。
- [`docs/codec.md`](../codec.md)：codec 统一离散张量接口与各 wrapper 的边界。
