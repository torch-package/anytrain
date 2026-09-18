# 局部职责重构（2026-09-18）

在当前工作区既有重构之上，完成 checkpoint、embedding、MoE 与 LongCat 四项后续整理。
公开组件入口、数值算法、参数注册路径及训练 checkpoint schema 保持兼容。

## 改动

- **Checkpoint**：callback 只提供触发 metadata、Trainer 序列化函数、logger 回调和跨 rank
  协调。storage 通过明确的方法提供临时文件准备、发布、删除、错误收集与关闭；操作阶段、
  audit 耗时和完成标记全部由 storage 更新。callback 不再调用 storage 内部方法或读写
  operation 字段。已有 backpressure、last 发布、故障回滚和 pickle 行为保留。
- **Embedding**：新增可选 `embedding_contract_state()` owner 方法，返回完整结构身份；
  显式 bank metadata 覆盖同名字段。契约生成会校验返回 mapping 和 canonical JSON 值域。
  旧 owner 属性推断集中在 `_embedding_contract.py` 的 legacy adapter，保留历史 payload
  与摘要；新 provider 不读取稠密 weight 或探测量化器属性，无需 provider registry。
- **MoE**：bank 保留唯一参数所有权、routing 校验、family 选择及 bias restore。linear/generic
  与 convolution 数值执行分别移动到无状态内部模块，参数以 tensor 显式传入。bank 与
  module-backed sparse 路径仅共享 host count 搬运和内存预算计算，独立保留 hooks 与未使用
  参数梯度语义。
- **LongCat**：`_loading.py` 拥有可选上游导入和 checkpoint 环境变量上下文，`_trace.py`
  拥有可选诊断记录；codec 保留 tensor、有效长度和 codebook 契约。原诊断字段与 CUDA
  synchronize/reset-peak 行为保留，并在模块文档说明测量影响。

故障注入测试跟随实际实现位置调整；新增 embedding provider、metadata 优先级、非法返回值、
动态身份更新，以及旧 dense/factorized payload 和 digest 兼容测试。

## 验证

使用 Conda `py312`：

- 相关测试：245 passed、3 skipped、1 deselected、62 subtests passed；该 deselected 项为
  后续单独复测的真实 Gloo 用例。
- 全量测试：沙箱内 1,599 passed、7 skipped、402 subtests passed；5 个真实 Gloo 用例在
  初始化回环网络时失败。相同五项在沙箱外复测全部通过（5 passed）。
- 本轮修改的 Python 文件与相关 MoE/LongCat 模块 Ruff 通过；`git diff --check` 通过。
- 全仓 basedpyright（显式指定 py312 Python）为 0 errors、1 warning。全仓 Ruff 的唯一
  错误和该 warning 均来自检查期间另外出现的 `audio/codec/glm4/features.py`：其 `__all__`
  引用尚未定义的 `glm4_pre_vq_tokenizer`。该并行文件不属于本轮改动，保持原状。

未运行 CUDA/NCCL 或下载真实 backend 权重。本记录区分本轮结果与工作区其它并行改动，
不将跨环境的两次测试表述为一次全量通过。
