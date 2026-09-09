# Agents

## 默认约定

- 开始前先阅读 `../AGENTS.md` 的默认约定，如果该文件存在；本文件只记录本仓库的补充约定。
- 本仓库使用 Conda 环境 `py312`（Python 3.12）；优先使用
  `conda run -n py312 <command>`，或先执行 `conda activate py312`，不要依赖系统
  Python。
- 提交前至少运行受影响测试，例如
  `python -m pytest -q tests/test_lightning_auto_cost.py`，并运行
  `python -m ruff check <changed-files>`。需要完整校验时运行
  `python -m pytest -q` 和 `python -m ruff check .`。

## 对外职责

- 本仓库承载通用 Lightning 训练框架契约（module / loss / evaluator）。新增核心能力前先在
  本仓 `todo.md` 确认有明确下游复用需求，不提前把 UTMOS、ASR 或 audio quality wrapper 放进 core。
