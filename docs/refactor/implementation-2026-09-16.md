# 模块审查修复记录（2026-09-16）

对应 [2026-09-15 审查](review-2026-09-15.md)，基线 `10f0991`。本轮修复全部四类已确认行为问题及类型诊断，完成报告执行顺序中第 2–4 项，并补充共享 LR 数学、PEFT 类型识别、Chat 生命周期和 S2S 依赖隔离。公开组件入口继续兼容；没有增加训练入口或领域能力。

## 行为与类型修复

| 问题 | 修复 | 回归证据 |
| --- | --- | --- |
| B1：cost 满队列、worker 失败时关闭卡死 | 有界轮询停止信号，持续检查 worker；总超时预算；失败后清理队列并传播原异常 | Event 控制竞态，确认关闭结束、异常传播及队列清空 |
| B2：FLOPs 测量破坏混合 train/eval 模式 | 保存并逐个恢复每个子模块的 training 标记 | 成功和 forward 异常两条路径 |
| B3：ASR 配置丢失文本评分语义 | text 拥有不可变 `TextMetricProfile`；ASR 完整转发 profile | 非默认 BLEU/chrF 配置往返、身份差异及历史默认兼容 |
| B4：mask 无法隔离非有限 padding | RL 在求和前屏蔽；flow 在相减、平方前屏蔽 prediction 和 target | NaN/Inf padding 不影响有效 loss，屏蔽项梯度为零 |
| 20 条原始类型诊断 | corpus 使用符合 Torch 接口的容器类型；PEFT 用 TypeGuard；spectral 明确 callable 边界；测试修正 mock 类型和 Optional 检查 | basedpyright 全仓 0 errors |

`WhisperSemanticConfig` 的历史 `bleu_tokenize="none"`、`chrf_word_order=0` profile 保持原 semantic payload 和 ID；其它 profile 增加完整 `text_profile` 身份字段。canonical JSON 的下沉保留原算法，未合并 `_semantic` 和 TTS hash 的不同值域，也未改变旧 digest。

## 组织与职责调整

| 模块 | 完成的调整 |
| --- | --- |
| 公共 contract / embedding | canonical serialization、hash 和 difference 下沉到 `_contract`；checkpoint 保留 state-dict schema 与兼容导出，embedding 不再依赖 Lightning |
| audit / ADT / quantization / evaluator | `_audit` 独立拥有 sink、context、事件及 collective tracing；`lightning.audit` 只拥有 hook adapter，公开入口和共享 sink 生命周期保持一致 |
| checkpoint callback | `_checkpoint_storage.AsyncCheckpointStorage` 独立拥有线程池、临时目录、backpressure、复制/原子发布、完成回调、审计和关闭；`_StorageJob` 统一 future 的 callbacks/publication/audit/source/target；移除无读取用途的 source 索引。callback 保留触发、top-k/last 状态和 rank 协调，以显式通知更新 Lightning 可见路径 |
| checkpoint 持久化 | storage 只提供 durable state 快照，拒绝仍有 job 的 pickle；恢复时重建线程和锁，并接受旧 callback 的空运行时字段。训练 checkpoint state-dict 没有换 schema |
| observation callback | `_observation_state` 集中拥有状态、delta、schema、快照、EMA 数学、历史编解码和用于 rank 预检的 schema signature；`_observation_writer` 消费快照。callback 保留 hook/cadence、collective 预检和错误同步；保留 v2/v3/v4 历史恢复 |
| materialization / service | `_rank_coordination` 统一逐 rank 协议、payload 校验和最低 rank 错误选择，adapter 保留各自生命周期及原诊断文本 |
| text / COMET | `_corpus` 统一 gather、rank-zero compute、错误 envelope、broadcast；各 evaluator 保留语义匹配、语料校验和评分。既有 envelope 字段与 audit source 名称保留 |
| TTS MOSS / Spark / CosyVoice | `_reference.ReferenceFiles` 统一临时文件生命周期和 encoded audio 写入；adapter 注入 waveform 保存函数，保留 CosyVoice mono 与 MOSS/Spark clamp、path alias 的不同语义及原临时文件名 |
| optim / schedule | `_lr_math` 提供纯 cosine interpolation，两个入口保留各自参数校验和时钟；`optim._adapter` 统一精确类型/FSDP 动态包装识别，不在普通 optimizer 导入时加载 PEFT |
| Chat | 增加 `close/aclose` 和同步/异步上下文管理器；关闭后拒绝文本请求，关闭幂等；async client 在所属 event loop 释放；cleanup 取消时保持锁直到 SDK 关闭结束。GLM 请求取消后等待实际后台调用结束，再允许后续请求或关闭 |
| S2S | `integration` 拥有领域 callbacks；condition/metrics 保留纯计算与 window；根包及旧 condition/metrics callback 路径通过 lazy export 兼容 |

独立子进程测试禁止导入 Lightning，验证 embedding、ADT、quantization、text、COMET、S2S 可独立导入。原有顺序、文件回滚、logger callback、DDP 错误同步和审计测试全部保留；对移动的私有实现，故障注入目标改为实际 owner。

## 逐模块覆盖与保留项

审查中的 29 项均已核对。1、3–10、13–15、17–18、24、26–29 的相关缺陷或明确共用职责已按上表处理。2、11–12、16、19–23、25 中未列出的部分，以及各模块内原本判断合理的边界保持原状。

报告明确以“继续增长 / 有后续功能需求 / 新增 backend 时”为前提的项目仍保留该前提：MoE family executor、BiCodec frontend、GLM4 source loader、LongCat trace、Qwen/translation backend、CosyVoice legacy config、quantizer EMA owner、BPE training 等没有仅因文件长度再切目录。Embedding 的量化 family metadata provider 也未新增通用 registry 或公开契约；当前没有第二套 provider 需求。这些是后续扩展方向，不是本轮确认的运行缺陷。

## 验证

使用 Conda `py312`。完整 pytest 包含真实 Gloo 本机多进程用例；这些用例需要沙箱外的本机通信权限。最终一次运行：**1,598 passed、7 skipped、399 subtests passed**（27.06 秒）；`ruff check .` 通过，basedpyright **0 errors / 0 warnings / 0 notes**，`git diff --check` 通过。

未下载真实 codec/TTS/ASR/COMET 权重，未执行 CUDA/NCCL 或真实模型长跑；本轮不声称通过这些环境验收。
