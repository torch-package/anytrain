# Agents

## 默认约定

- 开始前先阅读 `../AGENTS.md` 的默认约定，如果该文件存在；本文件只记录本仓库的补充约定。

## 对外职责

- 本仓库拥有 Speech 训练阶段 S3：冻结 codec 的 semantic→acoustic generator + GAN。
  S1/S2（codec 训练）归 `../anycodec`，阶段契约见 `../anycodec/docs/training-stages.md`；
  本仓库不重复实现 codec 训练路径。
- 对外交付 frame-aligned `AcousticGeneratorArtifact`（acoustic generator 预训练 + artifact
  导出契约）；`../speech-to-speech` 只消费该 artifact，不依赖本仓库内部实现。
- 本仓库同时承载通用 Lightning 训练框架契约（module / loss / evaluator）。新增核心能力前先在
  本仓 `todo.md` 确认有明确下游复用需求，不提前把 UTMOS、ASR 或 audio quality wrapper 放进 core。
