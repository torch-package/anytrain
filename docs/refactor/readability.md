# Readability Refactor

本文是模块级可读性重构的集中审阅入口。每个条目只记录当前代码已经落地并通过验证的
决策；公开行为的完整说明仍以 `docs/modules/` 中的对应模块文档为准。

## 原则

- 先确定公开契约和 ownership，再整理文件与命名。
- 保留已经有测试保护的数值、分布式、checkpoint 和 optional dependency 语义。
- 私有拆分必须对应真实职责，不能只为缩短文件而增加跳转。
- 每个模块独立验证并独立提交，便于集中阅读时按模块回退或比较。
- 新能力仍受根 `todo.md` 和仓库 `AGENTS.md` 的复用需求约束。

## `anytrain.loss`

状态：完成。

原问题：

- `_group.py` 与 `group.py` 共同拥有 loss 组合，名称和职责难以区分。
- `balancer.py` 同时包含数值实现、名称契约和完整 checkpoint schema 预检。
- loss-specific observation helper 与普通 loss API 一起从根包导出，用户入口与扩展 SPI
  混在同一层。

完成的调整：

- 合并 group ownership，使用 `_validate_loss_modules()`、`_resolve_balancer()` 和
  `_compute_group_loss()` 三个明确的私有操作服务普通与 spectral group。
- 抽出 `_balancer_contract.py` 和 `_balancer_checkpoint.py`；公开 balancer 类及导入路径
  保持不变。
- `anytrain.loss` 根入口只导出稳定 loss/balancer API；扩展实现显式使用
  `anytrain.loss.observation`。
- 更新 `docs/modules/loss.md`，使源码结构、公开入口、扩展 SPI 和验收命令与实现一致。

刻意保留：

- scalar-only loss protocol、所有 reduction 和 autograd 语义。
- balancer state-dict key、extra-state schema、严格预检和失败原子性。
- `LossABC` / `LossBalancerABC` 等现有公开名称。
- temporal、spectral 与 codec loss 的数值实现和 preset。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- `tests/test_loss.py`：86 passed，51 subtests passed。
