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

## `anytrain.evaluator`

状态：完成。

原问题：

- public `MetricDict` / `MetricValue` 定义在私有 `_validation.py`，部分实现又经由
  `abc.py` 间接导入，类型 ownership 不明确。
- evaluator name 与 metric key 的字符串规则分别实现，错误语义容易漂移。
- `weighted.py` 同时保留旧式 `TypeVar` 和 PEP 695 函数泛型，前者实际被遮蔽且无效。

完成的调整：

- 新增 public `types.py` 作为 metric 类型唯一 owner，package root 导入保持不变。
- core、text 和 speech evaluator 直接依赖 `types.py`，消除 `abc.py` 的隐藏 re-export。
- 名称校验统一归 `_validation.py`，`EvaluatorGroup` 只负责组合和 stateful lifecycle。
- 清理 weighted reduce 的无效泛型声明，并让 count cache 命名表达实际 key 语义。

刻意保留：

- `EvaluatorABC` 与 `EvaluatorGroup` 的现有公开名称和调用协议。
- mixed stateful/stateless group 在任何更新前失败的语义。
- text corpus/distributed reduction、fallback score 和中文静态映射。
- speech semantic config、模型加载、ASR 与 UTMOS backend 边界。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- evaluator core/text/speech：82 passed，9 subtests passed。

## `anytrain.observation`

状态：完成。

原问题：

- 全部 public value/spec 类型位于 `_types.py`，文件可见性与稳定公开职责不一致。
- `ForwardObservation` / `PollObservation` compatibility alias 在 package root 和类型实现文件
  重复定义，形成两个 owner。
- `_binding.py` 同时被 module author 和 Lightning adapter 使用，但 transport 方法的私有边界
  只存在于 `__all__`，模块文档没有说明。

完成的调整：

- 将 public value/spec owner 改为 `types.py`，package root 入口保持不变。
- compatibility alias 只在 package root 保留，且继续不进入 `__all__`。
- 文档明确 `types.py`、`_binding.py` 和 package root 的职责，以及 callback transport 的私有性。

刻意保留：

- exact concrete-type registry、一次性 recommendation hint 和 process-wide declaration。
- context propagation、event subscription、claim/release 与 compile no-op 语义。
- 所有 value/spec 校验、reduction、EMA 和 compatibility alias。
- Lightning `ObservationCallback` 的 distributed/checkpoint 行为。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- observation/event/Lightning integration：102 passed。

## `anytrain.checkpoint`

状态：完成。

审计结论：

- `contract.py` 与 `lightning.py` 已经分别拥有 canonical contract 和单文件恢复，继续拆分只会
  增加跳转。
- public 类型与函数均从 package root 显式导出，没有 optional dependency 或反向依赖。
- 唯一入口缺口是 loader 对 target module 与 hook 开关依赖隐式属性访问和真值转换。

完成的调整：

- `load_lightning_checkpoint()` 在读文件前严格校验 `nn.Module` target 和 bool hook 开关。
- 文档记录两个源码 owner、可信 checkpoint 前提和模块验收命令。

刻意保留：

- canonical JSON、SHA-256、first-difference 和 state-dict schema grammar。
- `weights_only=False`、CPU mapping、hook-before-state-load 和 `strict=True` 恢复顺序。
- 路径发现、Trainer resume、分片 checkpoint 和 metadata 解释仍由下游拥有。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- `tests/test_checkpoint.py`：14 passed，8 subtests passed。

## `anytrain.lightning`

状态：完成，不含单独审阅的 `schedule` 与 `perf`。

原问题：

- root 逐个 callback 文件维护导入，`callback/__init__.py` 又维护另一套导出，两处已经发生
  `ObservationCallback` 漂移。
- 没有测试保证两个 `__all__` 唯一、可解析或指向同一 callback 对象。
- `callback/observation.py` 很长，但其 state、checkpoint 和 distributed schema 共享 private
  不变量，单纯按行数拆分会创造新的内部协议。

完成的调整：

- callback package 成为 callback re-export 的唯一 owner，root 从 package 转发。
- `ObservationCallback` 新增 callback package 入口，原 root 入口保持不变。
- 新增 `tests/test_lightning_api.py`，约束导出完整性与对象 identity。
- 文档明确 capability、callback、materialization/service、schedule/perf 的 ownership。

刻意保留：

- 所有 callback 构造器、state key、checkpoint 和 Lightning hook 顺序。
- observation state machine、distributed preflight、EMA 与 writer 行为。
- async checkpoint、service、materialization、EMA、validation、OOM 和 logging 语义。
- schedule/perf 由后续独立模块条目处理。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- Lightning 本体与 observation 集成：184 passed，27 subtests passed。

## `anytrain.lightning.schedule`

状态：完成。

审计结论：

- curve、unit data、schedule spec、runtime compiler 和 unit callback 已有清晰 owner，不需要改动
  public API 或继续拆文件。
- unit callback 的 checkpoint loader 直接对 nested LR state 执行 `float(...)`，会接受字符串、
  bool、NaN 和畸形 group，并在完成验证前逐字段修改实例。
- 恢复的 base LR optimizer 数与 Trainer optimizer 数之间缺少显式检查。

完成的调整：

- 新增严格 nested LR group validator，只接受非空 sequence 与有限非负 host real。
- checkpoint state 完整验证后再统一赋值，失败不留下部分恢复结果。
- unit runtime 应用 LR 前检查 optimizer 数与 base LR group 数一致。
- 补 malformed LR/totals checkpoint 测试。

刻意保留：

- step-clock LambdaLR 编译、unit-clock provider 和 callback public API。
- accumulation、DDP all-gather、CUDA timing、throughput window、stop 与 log key 语义。
- schedule fingerprint 和现有 checkpoint 字段。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- `tests/test_lightning_schedule.py`：21 passed，30 subtests passed。

## `anytrain.lightning.perf`

状态：完成。

原问题：

- hardware、metrics 和 profile helper 各自复制正数校验，并通过 `float(...)` 接受字符串、bool
  和标量 Tensor，与 `PerformanceCallback` 的严格 host-real 契约不一致。
- 参数统计的过滤开关与设备名依赖隐式属性访问或真值转换，入口契约不明确。

完成的调整：

- 新增私有 `_validation.py`，统一 finite positive 与 non-negative host-real 校验。
- helper 和 `PerformanceCallback` 复用同一数值契约，错误类型与信息保持一致。
- 参数过滤开关与设备名在入口显式校验。
- 补 bool、字符串、Tensor 和控制字段的拒绝测试。

刻意保留：

- MFU 公式、窗口 ratio-of-sums、动态/静态/profile FLOPs 三种来源。
- 参数统计对 Lightning module 与轻量结构化适配对象的兼容性。
- optimizer update、梯度累积、DDP 聚合、CUDA 同步与有界 probe 语义。
- 硬件表、dtype alias、profiler 调用和全部 public 导入路径。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- `tests/test_perf.py`：27 passed，26 subtests passed。

## `anytrain.optim`

状态：完成。

审计结论：

- `_params.py` 已经是参数规划的唯一 owner，单次 module traversal 同时产出 Muon、AdamW decay、
  no-decay 和 LR scale 结果；继续拆分会把共享参数去重和最长规则匹配变成跨文件协议。
- `CompositeOptimizer`、LoRA-Muon 算法和 PEFT adapter 收集分别拥有独立状态/optional 边界，
  455 行的 LoRA-Muon 文件包含同一算法的数值核、状态和不变量，按长度拆分会降低可读性。

完成的调整：

- 参数规划显式校验 `requires_grad_only` 与 `decay_selected_params` 为 bool，避免隐式真值转换。
- `optim` 文档补充严格控制项、参数归属校验和“不按长度拆 LoRA-Muon”的 ownership 说明。

刻意保留：

- AdamW/Muon 分组规则、最长 LR scale 匹配、shared parameter 处理和 optional Muon/PEFT 边界。
- Composite optimizer 的名称化 checkpoint、scheduler param-group identity 与 closure 语义。
- LoRA-Muon 的 paper-faithful full-rank、momentum、atomic update 和 checkpoint 语义。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- `tests/test_optim.py` / `tests/test_optim_peft.py`：79 passed，3 subtests passed。

## `anytrain.module`

状态：完成，不含后续单独审阅的 `moe`、`idspace` 与 `stream`。

审计结论：

- 根包只转发 ADT 与 DiT 的稳定契约；quantization、Qwen、MoE、idspace 和 stream 保持显式子包，
  不会因 `import anytrain.module` 加载 optional Transformers。
- ADT 单文件共同维护统计 EMA、温度状态和 observation declaration；DiT 单文件共同维护 block、
  condition cache 与 trusted forward shape 契约，按行数拆分都会引入新的私有跨文件协议。
- quantization 与 Qwen 已有独立子包 owner 和完整模块文档，本轮不改它们已稳定的数值/optional 边界。

完成的调整：

- ADT config 与 forward 的统计开关改为严格 bool，拒绝隐式真值转换。
- DiT 的 `validate` 明确为严格 bool，同时保留 `validate=False` 的 trusted fast path。
- 模块文档补充这些控制项及根包/显式子包 ownership。

刻意保留：

- ADT 的统计、warmup、温度、checkpoint 与 observation 语义。
- DiT 的 condition modes、KV cache、mask、attention backend 和 feature projection 语义。
- quantization、Qwen 及各显式子包的 public import 边界。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- ADT / DiT tests：45 passed。
