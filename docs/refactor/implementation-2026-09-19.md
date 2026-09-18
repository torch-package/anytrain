# 全仓分析问题修复（2026-09-19）

本轮修复 [2026-09-18 分析](review-2026-09-18.md) 的 R1–R5，以及独立快照中的
UniCodec 类型检查差异。基于 `1496e7b`，不包含同时开发的 GLM4 features/shards。

## 行为变化

| 问题 | 修复 | 回归证据 |
| --- | --- | --- |
| R1：等奖励 advantage 为 NaN、整数 reward 失败 | reward 至少提升到 FP32、保留 FP64；三类 GRPO 统一 reward 诊断策略，ratio 至少用 FP32；零方差 norm 的反向梯度有限 | FP16/BF16/FP32/FP64/整数奖励，候选 mask，等奖励与非等奖励；完整 token/continuous/neighbor loss 与 FP64 对照并校验梯度 |
| R2：FP16 masked reduction 溢出 | mask 使用整数计数，分子至少 FP32 累积；masked MSE 在减法、平方前提升 dtype；各 reduction 保留原分母语义 | 65,536 有效项，flow/frame loss 为 0.25、均值为 1；三类 GRPO 的不同 reduction，以及 padding 的零梯度 |
| R3：Chat 关闭时等锁任务无法结束 | 保留共享 threading.Lock，异步等待改为非阻塞尝试与可取消 sleep，不再产生后台 acquire task/thread | 子进程内 asyncio.run 关闭有界、原锁持有者不变、释放后可重新获取；重复取消、同步/异步串行化和 SDK 生命周期回归 |
| R4：低精度权重破坏精确 count | 合法整数值 weights 经范围校验后用 int64 累计；总数支持 1 到 2**53 - 1，超范围显式报错 | BF16 count=257、FP16 count=65,536；共享 weights cache、加权均值与梯度；真实双进程 Gloo 跨 batch/rank 聚合 |
| R5：stream padding 污染输出 | source/target padding 用 masked_fill 隔离；LINEAR 屏蔽零权重分支；LINEAR/POOL/BROADCAST 至少 FP32 累积后转回 source dtype | 五种标准对齐的 NaN/Inf padding、零长度行、有效输出和梯度；65,536 项 FP16 pooling/broadcast |
| UniCodec：检查路径影响可选导入诊断 | 在 loader 边界通过 import_module 解析可选外部包；缺包或缺符号时保留安装指引 | 现有 mock backend/asset 测试、缺符号回归；独立快照 basedpyright 无错误 |

计算 dtype 的变化是公开数值行为的一部分：masked loss/GRPO/reduction 的 FP16/BF16
结果返回至少 FP32，FP64 保留；stream 的标准对齐保持 source 输出 dtype。计数精确不等于
浮点均值没有舍入误差。详见各模块文档中的数值和生命周期约定。

## 验收

使用 Conda `py312`。以 `git archive 1496e7b` 加本轮 src/tests 改动构建独立快照，
通过 `PYTHONPATH=<snapshot>/src` 检查，排除工作区其他开发和 editable import 路径影响。

- 全仓 `python -m pytest -q`：**1,680 passed、7 skipped、419 subtests passed**。
  已运行真实 Gloo 本机多进程测试。
- 随后新增低精度 count 的 Gloo 回归，并整理测试入口与源码换行；受影响的
  `test_lightning_validation.py`、`test_module_stream.py`、`test_unicodec_assets.py`
  再验证：**36 passed、23 subtests passed**，包含新增 Gloo 用例。
- 最终快照 `python -m ruff check .`：通过。
- 最终快照 `basedpyright --pythonpath /Users/zhuyin/miniconda3/envs/py312/bin/python`：
  **0 errors、0 warnings、0 notes**。
- 原始 `probes-2026-09-18.py` 再执行：等奖励 advantage/loss 为 0；masked mean 为 1，
  sequence mean 为 -1，flow/frame MSE 为 0.25；Metric 结果为 `(1, 257)` 与
  `(1, 65536)`；broadcast 为 `[3, 3]`；Chat 子进程正常退出。

本轮未运行 CUDA/NCCL 或下载真实可选后端权重；验证范围为 CPU 数值、真实本机 Gloo、
模拟 SDK 生命周期与独立静态检查。分析中的条件性架构建议和硬件长跑验收仍作为后续事项。
