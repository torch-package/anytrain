# Agents

## 默认约定

- 开始前先阅读 `../AGENTS.md` 的默认约定，如果该文件存在；本文件只记录本仓库的补充约定。

## 对外职责

- 本仓库承载通用 Lightning 训练框架契约（module / loss / evaluator）。新增核心能力前先在
  本仓 `todo.md` 确认有明确下游复用需求，不提前把 UTMOS、ASR 或 audio quality wrapper 放进 core。
