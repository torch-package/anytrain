# S2S Optional Domain

## 状态与结论

`anytrain.s2s` 第一版已经实现数据、模型能力和日志契约。它服务三类调用方：

| 能力 | S2S | Qwen-Omni | UniSS |
| --- | --- | --- | --- |
| 语义生成 `Generator` | 是 | 是 | 是 |
| token 单步 `TokenStepper` | 可选 | 否 | 可选 |
| teacher-forced `TokenScorer` | 可选 | 否 | 可选 |

三者真正共有的是语义层生成，不是模型内部结构。Qwen-Omni 是完整的
Thinker/Talker/Code2Wav 推理系统，第一版不要求它伪装成 token loop 或训练 scorer。S2S 与 UniSS
可以另外实现窄的 token 能力，但调用方仍通过同一组 Protocol 调用，不按模型名称分派。

## 边界

迁入 AnyTrain：

- text/audio 语义 request、response 和 reference sample；
- teacher-forced token batch 及 source audio condition；
- 高层生成、可选 token step、可选 token scoring 能力；
- modality loss 的精确 sum/count；
- 训练 step report、S2S 指标窗口和生成 latency/status 指标。

不迁入 AnyTrain：

- 具体 `nn.Module`、backbone、audio tower/head 和模型 factory；
- tokenizer、codec grammar、response program 和模型 prompt 模板；
- Hugging Face cache、audio-head cache 或 Qwen-Omni 内部输出类型；
- dataset adapter、collate、Hydra 配置、LightningModule 基类、CLI 和 job；
- ASR、UTMOS、codec decoder 等已经存在的第三方或 AnyTrain evaluator wrapper。

模型 provider 负责把自己的模型和上述契约适配起来。选择 provider 属于 composition，不属于
`anytrain.s2s` 运行时逻辑。

## 依赖

S2S 是 optional domain：

```toml
s2s = [
    "torch>=2.8",
    "liger-kernel>=0.8.1",
]
```

`liger-kernel` 是生产训练的必需依赖。AnyTrain 不复制 kernel、不提供 PyTorch fallback，也不包装
`LigerFusedLinearCrossEntropyLoss`。具体 scorer 直接调用上游 Liger，并返回标准化的 loss
sum/count。CPU contract test 使用 fake scorer，只验证接口，不代表 Liger CUDA smoke。

`import anytrain` 不导入 `anytrain.s2s`，package root 继续保持空且轻量。

## 数据契约

### 语义层

```python
Audio(waveform, sample_rate)
Content(text=None, audio=None)
Route(task, trace="direct")
GenerationRequest(source, route, response_modalities, target_language=None, sample_id=None)
GenerationResponse(text=None, audio=None)
Sample(request, target)
```

- waveform 固定为 `[channels, time]` 浮点 Tensor，sample rate 显式携带；
- `Content` 至少包含 text/audio 之一；
- `Route` 只承担 task/trace lineage 和稳定日志 tag，不解析下游 task program；
- request 显式声明期望的非空 output modality 集；
- reference target 可以是期望 modality 的非空子集。这样 Qwen-Omni 评测可以只有 text 或 audio
  reference，同时仍请求 text+audio 输出；
- 成功的 generation output 必须完整满足 request，带结构化 error 的 partial output 可以只包含
  request 的子集。

这里不接受 AnyDataset sample。下游 adapter 先把物理数据解释为上述语义对象，provider 不读取
workspace 路径或 dataset metadata。

### Token 训练层

训练数据分成模型输入和 lineage：

```python
TokenInput(input_ids, attention_mask, audio=None)
TokenTargets(labels, text_mask=None, audio_mask=None)
TokenBatch(input, targets, prompt_lengths, routes, sample_ids=None, audio_seconds=None)
```

`TokenScorer` 只收到 `TokenInput + TokenTargets`。`prompt_lengths`、route、sample ID 和 audio seconds
只用于校验与日志，不能泄漏给模型 provider。

source audio 有两个互斥且显式的表示：

```python
TokenAudioCondition(positions, mask)
EmbeddedAudioCondition(positions, mask, embeddings)
```

- UniSS 使用 `TokenAudioCondition`：codec payload 已在 `input_ids`，positions 标出模型内部需要处理的
  source audio token；
- 连续特征模型使用 `EmbeddedAudioCondition`：embedding 与 positions/mask 对齐；
- 两者共用唯一的 frame mask，inactive position 规范为 0；
- 不用一个 `embeddings=None` 的对象制造隐式变体。

token IDs、labels、position 和长度张量使用有符号整数；`uint8` 不属于契约，尤其不能承载
`IGNORE_INDEX=-100`。每个实际存在的 target modality mask 必须至少选择一个 token；没有该 modality
时传 `None`，不能用全 False mask 代替。loss 与日志中的离散计数统一为 `int64`，避免窗口累加时由
小整数 dtype 溢出。

公共对象在 CPU 构造时执行数值与跨字段校验；移动到 CUDA 后只做 dtype、shape、device 等不触发
host synchronization 的校验。生产热路径不执行 `bool(cuda_tensor)`。

## 模型能力

### 高层生成

所有三类 provider 实现同一个接口：

```python
class Generator(Protocol):
    def generate(
        self,
        requests: Sequence[GenerationRequest],
        options: GenerationOptions,
    ) -> Sequence[GenerationOutput]: ...
```

调用方统一使用 `generate(provider, requests, options)`。`GenerationOptions` 只包含三者共有的生成预算与
sampling 控制：`max_new_tokens`、`do_sample`、`temperature`、`top_p` 和 `seed`。Qwen-Omni 的 speaker、
Talker audio-token budget 等模型特有配置留在 provider，不扩张公共入口。公共 helper 只检查 row
alignment、返回类型和 modality 关系，不知道 provider 名称。provider 内部可以运行 constrained token
loop、官方 Qwen-Omni `generate()`，或 UniSS response program。

`GenerationOutput` 包含 response、可选 timing 和可选结构化 error。status 由这两个事实推导：

- response 且无 error：`succeeded`；
- response 且有 error：`partial`；
- 无 response 且有 error：`failed`。

stage timing 使用 provider 命名的开放 tuple，例如 `thinker`、`talker` 或 `decoder`；公共契约不固定
任何模型架构字段。

### 可选 token step

只有确实暴露 autoregressive token loop 的 provider 才实现：

```python
class TokenStepper(Protocol[StateT]):
    def step_tokens(
        self,
        input: TokenStepInput[StateT],
    ) -> TokenStepOutput[StateT]: ...
```

`TokenStepInput` 包含当前 input suffix、完整 context attention mask、显式且有序的
`selected_token_ids`、可选 prefill audio condition 和 provider-owned state。返回 logits 的最后一维
必须与 selected IDs 等长且顺序一致。

- cache prefill 使用完整 `input_ids`/attention mask、`state=None`，audio condition 只在这里传入；
- cached decode 使用短 suffix、覆盖已有 context 的更长 attention mask，并携带上一步 opaque state；
- uncached step 使用完整且等形的 `input_ids`/attention mask、`state=None`；
- `use_cache=True` 的输出必须返回 state，`use_cache=False` 的输出不得返回 state。

HF KV cache 与 audio-head cache 合成一个 opaque state；AnyTrain 不读取、拆分或重排它。需要 beam
cache selection 的 provider 在自身 adapter 内完成。Qwen-Omni 第一版不实现该能力。

这里故意只保留一种 vocabulary selection：显式 token IDs。调用方若需要整词表，就显式传入整组
IDs；不同时提供 modality、token kind 和 selected IDs 三套重叠入口。

### 可选 token scoring

```python
class TokenScorer(Protocol):
    def score_tokens(
        self,
        input: TokenInput,
        targets: TokenTargets,
    ) -> TokenLossStats: ...
```

scorer 自己负责 hidden state、routing、audio projection 和 Liger 调用。公共 objective 不访问模型成员，
只验证返回契约并计算：

```text
total.loss_sum = text.loss_sum + audio.loss_sum
total.count    = text.count    + audio.count
loss           = total.loss_sum / total.count
```

`total` 是派生值，不作为第三份可写事实保存。Qwen-Omni 当前是 inference-only，不实现 scorer。

## 统一调用方式

公共调用逻辑按所需能力接收 Protocol，而不是接收一个模型枚举：

```python
def evaluate(provider: Generator, requests):
    return generate(provider, requests)


def training_step(provider: TokenScorer, batch):
    return token_objective(provider, batch)
```

不存在 `if model == "qwen_omni"` 或 `getattr` fallback。一个入口需要 token step/scoring 时，它在装配
阶段就要求相应 Protocol；不具备能力的 provider 不进入该入口。

## 日志契约

`token_objective()` 返回标准 Lightning loss 和结构化 `StepReport`：

```python
{
    "loss": Tensor,
    "s2s_report": StepReport,
}
```

report 保存：

- global text/audio loss sum/count；
- 每个 route 的 row、active token、padding、text/audio target token 和 audio seconds；
- 仅在可精确归因时保存 route loss。

一个 batch 只有一个 route 时，整体 loss 可以归给该 route。混合 route batch 若 scorer 没有返回逐
route 充分统计，route loss 必须缺省，不能按 row 数拆分整体 mean。

精确性按整个日志窗口判定且不可逆：同一 route 在窗口内只要出现一次不可归因的 mixed-route
report，该窗口就不再输出这个 route 的 loss，即使窗口更早或更晚有 homogeneous batch。这个状态也
参与 DDP sum reduction，因此任一 rank 出现不可归因 report 都会关闭该 route 的窗口 loss。

`S2SMetricsCallback` 只消费 `StepReport`，不检查模型类型，也不重新解析 batch。它先累计加法量，再做
DDP sum reduction，最后计算 weighted loss、task/trace fraction、padding ratio 和 supervision ratio。
实际写入继续复用 `anytrain.lightning.experiment` 的 scalar capability。

DDP 要求各 rank 在相同训练 step 产生 `StepReport`，否则 callback 的 collective 无法对齐。callback
不持久化尚未 flush 的窗口；checkpoint 恢复后从新窗口开始，避免把 checkpoint 前后的不完整统计
拼成一个伪连续窗口。

`generation_metrics()` 记录 route、status、error code、total/stage seconds 和有输入音频时的 RTF。
throughput、MFU、optimizer、OOM、parameter/gradient observation 继续由 AnyTrain 通用 callback 负责，
S2S 不创建别名。

## 验证

contract tests 使用三个不加载权重的 fake provider：

- S2S、Qwen-Omni、UniSS 参数化执行同一条 semantic generation path；
- S2S、UniSS 参数化执行同一条 optional token-step/scoring path；
- Qwen-Omni 明确不实现 token-only capability；
- 覆盖 data shape/device/alignment、loss sum/count、混合 route 归因和 logger capability；
- package-root test 验证 `anytrain` 仍为空命名空间且 `anytrain.s2s` 导出可解析。

上线前仍需在目标 CUDA/Torch/Liger 组合上用真实 S2S/UniSS scorer 做数值和吞吐 smoke；该验证属于
具体 provider，不在 AnyTrain 内增加 Liger wrapper。
