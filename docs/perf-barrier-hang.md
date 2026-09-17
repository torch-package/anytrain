# PerformanceCallback per-batch barrier hang

## 结论

`PerformanceCallback.sync_distributed=True` 会在每个 train batch 的 `on_train_batch_start`
插入一次 `strategy.barrier`。这是一个没有超时的集合通信：只要有一个 rank 在当前 epoch 内
少运行一个 train batch（数据集重采样、变长或不均衡 loader、rank 局部提前结束），它离开训练
循环之后，其余 rank 就永久卡在该 barrier 上；NCCL 不报错，训练静默挂死。

该选项因此改为默认 `False`：它只服务测量口径，不应该有能力挂死训练。开启前提是各 rank 的
train batch 数严格相等，见 [`docs/modules/perf.md`](modules/perf.md)。

## 现场

- 触发任务：speech-to-speech `jobs/116` 的 `train/s2st/emilia_longcat_streaming_v3`
  8-GPU DDP 长跑。下游用 `SPEECH_TO_SPEECH_116_PERF_SYNC=0`
  （`callbacks.performance.sync_distributed=false`）绕过。
- 怀疑的上游触发源：`StreamingEpisodeDataset` 在 epoch 内重采样，使各 rank 的 batch 数
  分化，见 `speech-to-speech/jobs/116/11_emilia_streaming_bisect_2g_fixed_h100.yaml`。
- 同类先例：`speech-to-speech/docs/experiments/schedules/087-cost-rss-probe.md` 记录的
  "rank0 在 barrier、rank1 在 all_gather——集合通信永不配对"。

## 复旦最小复现

环境：`121`，2×A100，anytrain `10f0991`（修复前），Lightning 2.6.1，NCCL 2.27.5。

设置：2-rank DDP，rank0 比 rank1 多一个 train batch，
`PerformanceCallback(sync_cuda=True, sync_distributed=True)`。临时脚本放在
`/tmp/perf_sync_repro2.py`，未入库。

结果：rank1 `COMPLETED`；rank0 在 batch 120 永久阻塞，120s watchdog 打出的栈：

```text
performance.py:182  on_train_batch_start
performance.py:579  _sync_distributed
_capability.py:144  strategy_barrier
audit.py:488        audit_collective
lightning ddp.py:301 barrier
distributed_c10d.py:4881 barrier
```

## 未能确认的部分

同一合成设置下把 `sync_distributed` 关掉也会挂，但卡在 `performance.py:573 _sync_cuda`
（rank0 多出的那一步在做 DDP 梯度 allreduce 时等一个已经退出的 rank）。也就是"各 rank 迭代
数不一致"本身也会拖死 DDP。所以本复现确证的是 **"per-batch barrier 在 rank 进度分化时会
永久挂死"**，而不是下游那次分化的具体触发路径。下游观察到 `PERF_SYNC=0` 能跑，说明其分化
在 Lightning 循环层面仍能对齐，只有这个额外 barrier 配不上对。要坐实触发路径需要下游挂住
那次的栈或审计 JSONL。

## 处置

- `PerformanceCallback.sync_distributed` 默认改为 `False`，开启前提写入模块文档。
- 长跑建议开启 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 与 heartbeat timeout，让这类挂死变成
  明确报错，而不是静默阻塞。
- 曾尝试的 "barrier 前先 drain module device" 修复对本问题无效，已撤回；
  `auto_cost._synchronized_measurement` 的非 OOM 异常 rank 劈叉是独立缺陷，保留修复。
