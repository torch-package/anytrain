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

## `anytrain.module.moe`

状态：完成。

审计结论：

- `routing.py` 是 density-neutral record、mask 和 validation 的唯一 owner；`layer.py` 只编排
  partition/router/gate/bank/restore，公开入口没有重复状态。
- `bank.py` 的 dense、sparse、linear 与 conv 路径共享 additive bias、mask 和 restore 不变量；
  `expert.py` 的 stock Expert 共享 bind contract，按文件长度拆分会制造执行族之间的新协议。
- dynamic conv 的 shape、segment、partitioner、grouped kernels 和 router 已分别拥有清晰职责。

完成的调整：

- routing 的 `num_experts` / `top_k` validator 改为严格 Python integer，并统一拒绝 bool。
- router/gate 声明的 expert count 通过同一 validator；错误声明不再被当作有效整数。
- `LinearRouter.input_dim` 与 `Grid1d.segment_size` 对齐同一严格整数契约。

刻意保留：

- dense/top-k 统一 record、trusted internal ids 与 strict manual routing 的同步边界。
- masked execution、empty probe、additive bias、overlap-add coverage 和 mixed-dtype autograd 语义。
- Expert/Grid/partitioner ownership、stock expansion convenience 和 public import paths。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- MoE/routing/dynamic-conv tests：83 passed，1 skipped，29 subtests passed。

## `anytrain.module.idspace`

状态：完成。

审计结论：

- `layout.py` 唯一拥有命名半开区间和 local/global ID 转换；`embedding.py` 唯一拥有按 block
  路由、adapter 与 trusted selected-block fast path，两个文件无需新增中间层。
- homogeneous inference 与通用训练路径的分离是为避免未命中参数进入 autograd，不是重复实现。

完成的调整：

- Layout block 端点、单 ID 与 ignore 改为严格 integer；转换入口落实 Tensor-only 契约。
- Embedding 显式校验 Layout、block embedding、input tensor、strict bool validate 和 block-name
  collection，错误由实际 owner 在注册/执行前暴露。

刻意保留：

- 不连续 global space、同 block local conversion 与 ignore passthrough 语义。
- homogeneous no-sync fast path、未命中参数 `grad is None` 和 trusted preselected path。
- dtype 错误继续由 PyTorch embedding 暴露，adapter/output-head 仍由下游拥有。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- idspace layout/embedding tests：25 passed，4 subtests passed。

## `anytrain.module.stream`

状态：完成。

审计结论：

- expression、layout、binding、materialization 和 output 已按声明、离散几何、payload、连续执行、
  target labels 分成五个明确 owner；不存在需要合并的重复抽象。
- `layout.py` 的 DAG compilation 与 `materialization.py` 的 node evaluation 各自维护完整遍历状态，
  再按长度拆分会把 cycle/cache/lineage 或 embed/align/flatten cache 变成跨文件协议。

完成的调整：

- expression tuple 容器、timeline/payload/module/target Mapping 在对应 public stage 入口校验。
- custom operator 在 Materializer 构造时校验为 `AlignOperator`，不再延迟到实际 Align 节点执行。
- control id 和 predicted-control collection 区分类型错误与范围/未知名称错误。

刻意保留：

- carrier/source lineage、align geometry、nested flatten、control namespace 与 local target 语义。
- DAG 节点只 materialize 一次、分批 bind 和 layout-before-payload 的工作流。
- codec inventory、global token ID、生成 grammar 和 output dependency 继续由下游拥有。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- `tests/test_module_stream.py`：13 passed。

## `anytrain.audio`

状态：完成，不含后续单独审阅的 `bpe` 与 `codec`。

审计结论：

- 根包为空导出并保持轻量；tokenizer/detokenizer 子包分别转发 directional ABC 与 lazy codec
  loader，不会因 `import anytrain.audio` 加载 backend。
- tokenizer 要求完整 Tensor inventory，detokenizer 允许 per-stream `None`，两套校验语义不同，
  不应为了去重抽成一个带分支的共享 helper。

完成的调整：

- `PaddedAudioCodes` 显式要求每条 stream 的 codes/lengths 同 device，并要求所有 stream 同 device。
- 文档补充 directional validation ownership 与 padded batch/device contract。

刻意保留：

- tokenizer 的完整 stream inventory、signed IDs、vocab range 与 batch 对齐语义。
- detokenizer 的 optional stream features 和 code-to-feature-to-waveform convenience path。
- backend registry/cache、sample-rate adaptation 与 waveform validation 继续由 `audio.codec` 拥有。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- audio directional tests：40 passed。

## `anytrain.audio.bpe`

状态：完成。

审计结论：

- `_frame.py`、`_core.py`、`api.py`、`trainer.py` 与 `_eval.py` 分别拥有 frame alphabet、
  BPE state/training bridge、artifact workflow、trainer identity 和 evaluation stats。
- optional `tokenizers` import 只在构造/加载/训练时发生；package root 保持轻量。
- progress/alphabet scan 与 max-frame 完整 sequence 边界已有 regression/golden 覆盖，不需要重写。

完成的调整：

- codebook size、frame/base id、BPE token id 与 eval `top_k` 对齐严格 integer 契约。
- `train_from_spec()` 在读取 corpus 或加载 trainer runtime 前显式要求 fresh-iterable callable。
- public `show_progress` 明确为 bool，不再接受隐式真值。

刻意保留：

- mixed-radix order、private-use alphabet、tokenizers BPE merges 和 compact token ids。
- trainer semantic identity、single/multi-codebook scan count、progress 与 max_frames 行为。
- `codec_bpe.json` schema、encode/decode 和完整 evaluation statistics。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- BPE unit/regression tests：49 passed，24 subtests passed。

## `anytrain.audio.codec`

状态：完成。

原问题：

- public `AudioPlanningEstimator` 定义在私有 `_planning.py`，公开类型与算法实现 owner 混合。
- `CodecEntry` 只校验 capability/factory 的 presence，对 name、spec、identity preset 与 callable
  factory 依赖后续隐式失败。
- cache 下载控制项依赖隐式真值，FSQ capability 通过 `int()` / `float()` 接受 bool、float 或字符串。

完成的调整：

- 新增 public `planning.py` 作为 estimator protocol owner，`_planning.py` 只保留静态算法与 identity。
- registry descriptor 在构造时完整校验静态结构；public preset 查询统一要求非空字符串名称。
- Hugging Face 下载控制项严格要求 bool；FSQ levels/value metadata 先校验 tuple、integer 与 host-real
  结构，再做范围和对齐检查。
- `docs/codec.md` 记录 spec/planning/registry/cache/concrete backend 的 ownership 与验收范围。

刻意保留：

- tokenizer/detokenizer directional ABC、named stream inventory 与 adapter 行为。
- identity/device backend cache、pinned revision、planning hash 和长度估算公式。
- BiCodec、DAC、GLM-4、LongCat、Stable Codec 与 UniCodec 的模型加载、optional import、原生 tensor
  layout 和数值实现；这些 backend 内部各自是一套 upstream adapter，不按文件长度机械拆分。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- codec core/cache：39 passed。
- concrete backend/assets：108 passed。

## `anytrain.tts`

状态：完成。

审计结论：

- `protocol.py` 已经集中拥有 reference/capability/options/output/backend public contract，不需要按类型
  数量拆文件。
- MOSS 已按 config、optional loading、runtime/reference lifecycle、output normalization 拆成真实职责；
  Qwen/CosyVoice/Spark 的 conditioning 和 upstream inference 协议不同，不应抽一个分支繁多的基类。

完成的调整：

- 公共 option/output 统一复用 strict positive real/probability/string/mapping validator。
- `language`、`temperature`、`top_p`、`duration`、`meta` 与 `waveform_duration()` 显式校验类型、有限性
  和非空 shape，不再依赖比较运算或隐式属性错误。
- MOSS config/loader 控制项严格校验字符串和 bool；processor sample rate、output duration/sample rate
  不再通过 `int()` / `float()` 接受字符串或 bool。
- `docs/tts.md` 记录 public/shared/provider ownership 与各 backend 的保留边界。

刻意保留：

- 单条/批量返回类型、explicit-none option merge、voice capability 与 speaker inventory 语义。
- in-memory/path reference 的 request-scoped materialization 和 compatibility alias。
- MOSS processor generation、Qwen CustomVoice、CosyVoice vLLM concurrency 与 Spark sequential batch 的
  upstream-specific 行为和 optional dependencies。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- TTS public contract 与四个 backend tests：69 passed。

## `anytrain.chat`

状态：完成。

审计结论：

- public types、config/env、client construction、request lifecycle、response normalization、image HTTP
  与 OpenRouter catalog 已有明确 owner，不需要继续拆分。
- `_request.py` 的同步/异步 provider payload 与日志共享 completion spec 和完整生命周期，按 provider
  拆开反而会复制 logging/response contract。

完成的调整：

- `ChatConfig` 在创建 SDK client 或发送 catalog 凭据前校验 endpoint/key、primary-first unique model
  inventory、allowlist 与 candidate limit。
- 四个文本入口统一严格校验非空字符串 prompt 和 bool `refresh`；image iterable 元素与有限正 timeout
  在文件读取/HTTP 前校验。
- OpenRouter catalog 明确拒绝 NaN/Infinity pricing，同时保留 JSON numeric string 解析。
- 模块文档补充 public/client/request/response/image/catalog ownership 与验收范围。

刻意保留：

- 实例内成功消息历史、refresh、sync/async serialization 与跨 event-loop 拒绝语义。
- DeepSeek/GLM/OpenRouter payload、optional SDK import、请求日志和 response metadata。
- OpenRouter live catalog 排序/fallback 与 GLM image 独立无上下文请求。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- Chat API/config/image/OpenRouter/provider tests：51 passed。

## `anytrain.framework`

状态：完成。

审计结论：

- flow matching、GAN 和 RL 已按各自的 source/runtime/objective、output/loss/preset、rollout/loss
  ownership 组织，不需要按文件长度继续拆分。
- 三套算法的生命周期、张量布局和 reduction contract 不同，抽取共同基类会把分支条件移入抽象层，
  降低局部可读性。

完成的调整：

- 新增私有 `_validation.py` 作为 framework host scalar contract 的唯一 owner。
- flow source/time/sampler、GAN 权重、RL loss 超参数统一拒绝 bool、Tensor、字符串和非有限实数，
  各算法文件只表达自身范围约束。
- RL `validate`、DPO `reference_free` 与 sampler bool 控制项改为严格 bool，不再依赖隐式真值。
- 模块文档补充共享校验 ownership、三套算法的保留边界与 trusted-path 语义。

刻意保留：

- flow matching 的 optional dependency、source/path/runtime/objective/sampler 组合与数值行为。
- GAN output normalization、feature matching、gradient penalty、observation event 和 DAC preset 架构。
- DPO、token/continuous/neighbor GRPO 的公式、mask/reduction、policy-only 梯度与
  `validate=False` 无 tensor host synchronization 快速路径。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- framework flow matching、GAN、RL tests：88 passed，1 skipped，38 subtests passed。

## `anytrain.translation`

状态：完成。

审计结论：

- `protocol.py` 的 dataset-independent batch contract 与 Qwen backend lifecycle 已有明确边界。
- Qwen spec、load policy、factory、token estimator、prompt/decoder 和 runtime 共同定义一套 pinned
  semantic identity，按文件长度拆开会增加跳转且没有产生新的独立 owner。

完成的调整：

- 新增私有 `_validation.py`，统一拥有非空字符串、字符串序列与正整数契约；protocol 和 Qwen 不再
  分别维护同类 helper。
- spec/load/factory 明确拒绝 bool integer、非字符串 dtype/protocol、空 cache/device 字符串。
- Qwen runtime 在第三方调用边界校验 callable、mapping、rank-2 integer token ids、batch alignment、
  prompt-prefix length 与 decoded output，不再泄漏 `KeyError`、`AttributeError` 或字符序列隐式行为。
- 翻译文档和模块索引补充 ownership、保留边界和验收范围。

刻意保留：

- `UniversalTranslator` 的 position-aligned `Sequence[str] -> list[str]` 协议。
- model revision、generation options、prompt/decoder protocol 构成的 semantic id 与 config hash。
- physical load options 不进入 identity、pickle-safe lazy factory、tokenizer-only estimator 与
  local-only snapshot resolution。
- deterministic generation、left padding、thinking marker 清理和 optional Transformers import。

验收：

- `ruff`：通过。
- `basedpyright`：0 errors。
- translation tests：13 passed，13 subtests passed。
